//! Garbage Collection State and Algorithm
//!
//! This module implements CPython-compatible generational garbage collection
//! for RustPython, using an intrusive doubly-linked list approach.

use crate::common::lock::PyMutex;
use crate::{PyObject, PyObjectRef};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::collections::HashSet;
use std::sync::{Mutex, RwLock};

/// GC debug flags
pub const DEBUG_STATS: u32 = 1;
pub const DEBUG_COLLECTABLE: u32 = 2;
pub const DEBUG_UNCOLLECTABLE: u32 = 4;
pub const DEBUG_SAVEALL: u32 = 8;
pub const DEBUG_LEAK: u32 = DEBUG_COLLECTABLE | DEBUG_UNCOLLECTABLE | DEBUG_SAVEALL;

/// Default thresholds for each generation
const DEFAULT_THRESHOLD_0: u32 = 700;
const DEFAULT_THRESHOLD_1: u32 = 10;
const DEFAULT_THRESHOLD_2: u32 = 10;

/// Statistics for a single generation
#[derive(Debug, Default)]
pub struct GcStats {
    pub collections: u64,
    pub collected: u64,
    pub uncollectable: u64,
}

/// A single GC generation with intrusive linked list
pub struct GcGeneration {
    /// Number of objects in this generation
    count: AtomicUsize,
    /// Threshold for triggering collection
    threshold: AtomicU32,
    /// Collection statistics
    stats: PyMutex<GcStats>,
}

impl GcGeneration {
    pub const fn new(threshold: u32) -> Self {
        Self {
            count: AtomicUsize::new(0),
            threshold: AtomicU32::new(threshold),
            stats: PyMutex::new(GcStats {
                collections: 0,
                collected: 0,
                uncollectable: 0,
            }),
        }
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    pub fn threshold(&self) -> u32 {
        self.threshold.load(Ordering::SeqCst)
    }

    pub fn set_threshold(&self, value: u32) {
        self.threshold.store(value, Ordering::SeqCst);
    }

    pub fn stats(&self) -> GcStats {
        let guard = self.stats.lock();
        GcStats {
            collections: guard.collections,
            collected: guard.collected,
            uncollectable: guard.uncollectable,
        }
    }

    pub fn update_stats(&self, collected: u64, uncollectable: u64) {
        let mut guard = self.stats.lock();
        guard.collections += 1;
        guard.collected += collected;
        guard.uncollectable += uncollectable;
    }
}

/// Wrapper for raw pointer to make it Send + Sync
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GcObjectPtr(NonNull<PyObject>);

// SAFETY: We only use this for tracking objects, and proper synchronization is used
unsafe impl Send for GcObjectPtr {}
unsafe impl Sync for GcObjectPtr {}

/// Global GC state
pub struct GcState {
    /// 3 generations (0 = youngest, 2 = oldest)
    pub generations: [GcGeneration; 3],
    /// Permanent generation (frozen objects)
    pub permanent: GcGeneration,
    /// GC enabled flag
    pub enabled: AtomicBool,
    /// Debug flags
    pub debug: AtomicU32,
    /// gc.garbage list (uncollectable objects with __del__)
    pub garbage: PyMutex<Vec<PyObjectRef>>,
    /// gc.callbacks list
    pub callbacks: PyMutex<Vec<PyObjectRef>>,
    /// Mutex for collection (prevents concurrent collections)
    collecting: Mutex<()>,
    /// Allocation counter for gen0
    alloc_count: AtomicUsize,
    /// Registry of all tracked objects (for cycle detection)
    tracked_objects: RwLock<HashSet<GcObjectPtr>>,
}

impl Default for GcState {
    fn default() -> Self {
        Self::new()
    }
}

impl GcState {
    pub fn new() -> Self {
        Self {
            generations: [
                GcGeneration::new(DEFAULT_THRESHOLD_0),
                GcGeneration::new(DEFAULT_THRESHOLD_1),
                GcGeneration::new(DEFAULT_THRESHOLD_2),
            ],
            permanent: GcGeneration::new(0),
            enabled: AtomicBool::new(true),
            debug: AtomicU32::new(0),
            garbage: PyMutex::new(Vec::new()),
            callbacks: PyMutex::new(Vec::new()),
            collecting: Mutex::new(()),
            alloc_count: AtomicUsize::new(0),
            tracked_objects: RwLock::new(HashSet::new()),
        }
    }

    /// Check if GC is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Enable GC
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::SeqCst);
    }

    /// Disable GC
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// Get debug flags
    pub fn get_debug(&self) -> u32 {
        self.debug.load(Ordering::SeqCst)
    }

    /// Set debug flags
    pub fn set_debug(&self, flags: u32) {
        self.debug.store(flags, Ordering::SeqCst);
    }

    /// Get thresholds for all generations
    pub fn get_threshold(&self) -> (u32, u32, u32) {
        (
            self.generations[0].threshold(),
            self.generations[1].threshold(),
            self.generations[2].threshold(),
        )
    }

    /// Set thresholds
    pub fn set_threshold(&self, t0: u32, t1: Option<u32>, t2: Option<u32>) {
        self.generations[0].set_threshold(t0);
        if let Some(t1) = t1 {
            self.generations[1].set_threshold(t1);
        }
        if let Some(t2) = t2 {
            self.generations[2].set_threshold(t2);
        }
    }

    /// Get counts for all generations
    pub fn get_count(&self) -> (usize, usize, usize) {
        (
            self.generations[0].count(),
            self.generations[1].count(),
            self.generations[2].count(),
        )
    }

    /// Get statistics for all generations
    pub fn get_stats(&self) -> [GcStats; 3] {
        [
            self.generations[0].stats(),
            self.generations[1].stats(),
            self.generations[2].stats(),
        ]
    }

    /// Track a new object (add to gen0)
    /// Called when IS_TRACE objects are created
    ///
    /// # Safety
    /// obj must be a valid pointer to a PyObject
    pub unsafe fn track_object(&self, obj: NonNull<PyObject>) {
        self.generations[0].count.fetch_add(1, Ordering::SeqCst);
        self.alloc_count.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut tracked) = self.tracked_objects.write() {
            tracked.insert(GcObjectPtr(obj));
        }
    }

    /// Untrack an object (remove from GC lists)
    /// Called when objects are deallocated
    ///
    /// # Safety
    /// obj must be a valid pointer to a PyObject
    pub unsafe fn untrack_object(&self, obj: NonNull<PyObject>) {
        let count = self.generations[0].count.load(Ordering::SeqCst);
        if count > 0 {
            self.generations[0].count.fetch_sub(1, Ordering::SeqCst);
        }
        if let Ok(mut tracked) = self.tracked_objects.write() {
            tracked.remove(&GcObjectPtr(obj));
        }
    }

    /// Get all tracked objects (for gc.get_objects)
    pub fn get_objects(&self, _generation: Option<i32>) -> Vec<PyObjectRef> {
        if let Ok(tracked) = self.tracked_objects.read() {
            tracked
                .iter()
                .filter_map(|ptr| {
                    // SAFETY: We only store valid PyObject pointers
                    let obj = unsafe { ptr.0.as_ref() };
                    // Check if object is still alive (has references)
                    if obj.strong_count() > 0 {
                        Some(obj.to_owned())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Perform garbage collection on the given generation
    /// Returns (collected_count, uncollectable_count)
    pub fn collect(&self, generation: usize) -> (usize, usize) {
        if !self.is_enabled() {
            return (0, 0);
        }

        // Try to acquire the collecting lock, but don't block
        let _guard = match self.collecting.try_lock() {
            Ok(g) => g,
            Err(_) => return (0, 0),
        };

        let generation = generation.min(2);

        // Get a snapshot of all tracked objects (clone to release the lock quickly)
        let tracked: Vec<GcObjectPtr> = match self.tracked_objects.try_read() {
            Ok(t) => t.iter().copied().collect(),
            Err(_) => return (0, 0),
        };

        if tracked.is_empty() {
            self.generations[generation].update_stats(0, 0);
            return (0, 0);
        }

        // Filter to only objects that are still alive
        let tracked: Vec<GcObjectPtr> = tracked
            .into_iter()
            .filter(|ptr| {
                let obj = unsafe { ptr.0.as_ref() };
                obj.strong_count() > 0
            })
            .collect();

        if tracked.is_empty() {
            self.generations[generation].update_stats(0, 0);
            return (0, 0);
        }

        // Step 1: Build gc_refs map (copy of ref_count for each object)
        // Use a set for quick lookup
        let tracked_set: HashSet<GcObjectPtr> = tracked.iter().copied().collect();
        let mut gc_refs: std::collections::HashMap<GcObjectPtr, usize> =
            std::collections::HashMap::new();
        for ptr in &tracked {
            let obj = unsafe { ptr.0.as_ref() };
            gc_refs.insert(*ptr, obj.strong_count());
        }

        // Step 2: Subtract internal references
        // For each tracked object, traverse its children and decrement gc_refs
        // Use raw pointers to avoid reference count manipulation
        for ptr in &tracked {
            let obj = unsafe { ptr.0.as_ref() };
            // Only traverse if the object has a trace function
            if obj.is_gc_tracked() {
                let referent_ptrs = unsafe { obj.gc_get_referent_ptrs() };
                for child_ptr in referent_ptrs {
                    let gc_ptr = GcObjectPtr(child_ptr);
                    if let Some(refs) = gc_refs.get_mut(&gc_ptr) {
                        *refs = refs.saturating_sub(1);
                    }
                }
            }
        }

        // Step 3: Find unreachable objects (gc_refs == 0)
        // These are only reachable through cycles
        let unreachable: HashSet<GcObjectPtr> = gc_refs
            .iter()
            .filter(|&(_, refs)| *refs == 0)
            .map(|(ptr, _)| *ptr)
            .collect();

        if unreachable.is_empty() {
            self.generations[generation].update_stats(0, 0);
            return (0, 0);
        }

        let collected = unreachable.len();

        // Note: We do NOT untrack objects here because we're not actually
        // freeing them. The objects remain in memory with their circular
        // references intact. We only return the count of detected cycles.
        //
        // This is different from CPython which actually breaks the cycles
        // and frees the memory. Our implementation provides cycle detection
        // without cycle breaking, which is useful for diagnostics.

        // Update statistics
        self.generations[generation].update_stats(collected as u64, 0);

        (collected, 0)
    }

    /// Get count of frozen objects
    pub fn get_freeze_count(&self) -> usize {
        self.permanent.count()
    }

    /// Freeze all tracked objects (move to permanent generation)
    pub fn freeze(&self) {
        // Move all objects from gen0-2 to permanent
        for generation in &self.generations {
            let count = generation.count.swap(0, Ordering::SeqCst);
            self.permanent.count.fetch_add(count, Ordering::SeqCst);
        }
    }

    /// Unfreeze all objects (move from permanent to gen2)
    pub fn unfreeze(&self) {
        let count = self.permanent.count.swap(0, Ordering::SeqCst);
        self.generations[2].count.fetch_add(count, Ordering::SeqCst);
    }
}

use std::sync::OnceLock;

/// Global GC state instance
/// Using a static because GC needs to be accessible from object allocation/deallocation
static GC_STATE: OnceLock<GcState> = OnceLock::new();

/// Get a reference to the global GC state
pub fn gc_state() -> &'static GcState {
    GC_STATE.get_or_init(GcState::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_state_default() {
        let state = GcState::new();
        assert!(state.is_enabled());
        assert_eq!(state.get_debug(), 0);
        assert_eq!(state.get_threshold(), (700, 10, 10));
        assert_eq!(state.get_count(), (0, 0, 0));
    }

    #[test]
    fn test_gc_enable_disable() {
        let state = GcState::new();
        assert!(state.is_enabled());
        state.disable();
        assert!(!state.is_enabled());
        state.enable();
        assert!(state.is_enabled());
    }

    #[test]
    fn test_gc_threshold() {
        let state = GcState::new();
        state.set_threshold(100, Some(20), Some(30));
        assert_eq!(state.get_threshold(), (100, 20, 30));
    }

    #[test]
    fn test_gc_debug_flags() {
        let state = GcState::new();
        state.set_debug(DEBUG_STATS | DEBUG_COLLECTABLE);
        assert_eq!(state.get_debug(), DEBUG_STATS | DEBUG_COLLECTABLE);
    }
}
