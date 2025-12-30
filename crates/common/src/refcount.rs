//! Reference counting implementation based on circ's EBR (Epoch-Based Reclamation).
//!
//! This module provides a RefCount type that is compatible with circ's memory reclamation
//! system while maintaining the original API for backward compatibility.

use std::sync::atomic::{AtomicU64, Ordering};

// Re-export circ types for use in vm
pub use circ::utils::{
    COUNT, DESTRUCTED, Deferable, EPOCH, EPOCH_MASK_HEIGHT, EPOCH_WIDTH, Modular, RcInner, STRONG,
    STRONG_WIDTH, State, WEAK, WEAK_COUNT, WEAK_WIDTH, WEAKED,
};
pub use circ::{Guard, cs, global_epoch};

/// LEAKED bit for interned objects (never deallocated)
/// Position: just after WEAKED bit
pub const LEAKED: u64 = 1 << (EPOCH_MASK_HEIGHT - 3);

/// Extended State with LEAKED support
#[derive(Clone, Copy)]
pub struct PyState {
    inner: u64,
}

impl PyState {
    #[inline]
    pub fn from_raw(inner: u64) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn as_raw(self) -> u64 {
        self.inner
    }

    #[inline]
    pub fn strong(self) -> u32 {
        ((self.inner & STRONG) / COUNT) as u32
    }

    #[inline]
    pub fn weak(self) -> u32 {
        ((self.inner & WEAK) / WEAK_COUNT) as u32
    }

    #[inline]
    pub fn destructed(self) -> bool {
        (self.inner & DESTRUCTED) != 0
    }

    #[inline]
    pub fn weaked(self) -> bool {
        (self.inner & WEAKED) != 0
    }

    #[inline]
    pub fn leaked(self) -> bool {
        (self.inner & LEAKED) != 0
    }

    #[inline]
    pub fn epoch(self) -> u32 {
        ((self.inner & EPOCH) >> EPOCH_MASK_HEIGHT) as u32
    }

    #[inline]
    pub fn with_epoch(self, epoch: usize) -> Self {
        Self::from_raw((self.inner & !EPOCH) | (((epoch as u64) << EPOCH_MASK_HEIGHT) & EPOCH))
    }

    #[inline]
    pub fn add_strong(self, val: u32) -> Self {
        Self::from_raw(self.inner + (val as u64) * COUNT)
    }

    #[inline]
    pub fn sub_strong(self, val: u32) -> Self {
        debug_assert!(self.strong() >= val);
        Self::from_raw(self.inner - (val as u64) * COUNT)
    }

    #[inline]
    pub fn add_weak(self, val: u32) -> Self {
        Self::from_raw(self.inner + (val as u64) * WEAK_COUNT)
    }

    #[inline]
    pub fn with_destructed(self, dest: bool) -> Self {
        Self::from_raw((self.inner & !DESTRUCTED) | if dest { DESTRUCTED } else { 0 })
    }

    #[inline]
    pub fn with_weaked(self, weaked: bool) -> Self {
        Self::from_raw((self.inner & !WEAKED) | if weaked { WEAKED } else { 0 })
    }

    #[inline]
    pub fn with_leaked(self, leaked: bool) -> Self {
        Self::from_raw((self.inner & !LEAKED) | if leaked { LEAKED } else { 0 })
    }
}

/// Reference count using circ's state layout with LEAKED support.
///
/// State layout (64 bits):
/// [4 bits: epoch] [1 bit: destructed] [1 bit: weaked] [1 bit: leaked] [28 bits: weak_count] [29 bits: strong_count]
pub struct RefCount {
    state: AtomicU64,
}

impl Default for RefCount {
    fn default() -> Self {
        Self::new()
    }
}

impl RefCount {
    /// Create a new RefCount with strong count = 1
    pub fn new() -> Self {
        // Initial state: strong=1, weak=1 (implicit weak for strong refs)
        Self {
            state: AtomicU64::new(COUNT + WEAK_COUNT),
        }
    }

    /// Get current strong count
    #[inline]
    pub fn get(&self) -> usize {
        PyState::from_raw(self.state.load(Ordering::SeqCst)).strong() as usize
    }

    /// Increment strong count
    #[inline]
    pub fn inc(&self) {
        let val = PyState::from_raw(self.state.fetch_add(COUNT, Ordering::SeqCst));
        if val.destructed() {
            // Already marked for destruction, but we're incrementing
            // This shouldn't happen in normal usage
            std::process::abort();
        }
        if val.strong() == 0 {
            // The previous fetch_add created a permission to run decrement again
            self.state.fetch_add(COUNT, Ordering::SeqCst);
        }
    }

    /// Try to increment strong count. Returns true if successful.
    /// Returns false if the object is already being destructed.
    #[inline]
    pub fn safe_inc(&self) -> bool {
        let mut old = PyState::from_raw(self.state.load(Ordering::SeqCst));
        loop {
            if old.destructed() {
                return false;
            }
            let new_state = old.add_strong(1);
            match self.state.compare_exchange(
                old.as_raw(),
                new_state.as_raw(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(curr) => old = PyState::from_raw(curr),
            }
        }
    }

    /// Decrement strong count. Returns true when count drops to 0.
    #[inline]
    pub fn dec(&self) -> bool {
        let old = PyState::from_raw(self.state.fetch_sub(COUNT, Ordering::SeqCst));

        // LEAKED objects never reach 0
        if old.leaked() {
            return false;
        }

        old.strong() == 1
    }

    /// Mark this object as leaked (interned). It will never be deallocated.
    pub fn leak(&self) {
        debug_assert!(!self.is_leaked());
        let mut old = PyState::from_raw(self.state.load(Ordering::SeqCst));
        loop {
            let new_state = old.with_leaked(true);
            match self.state.compare_exchange(
                old.as_raw(),
                new_state.as_raw(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return,
                Err(curr) => old = PyState::from_raw(curr),
            }
        }
    }

    /// Check if this object is leaked (interned).
    pub fn is_leaked(&self) -> bool {
        PyState::from_raw(self.state.load(Ordering::Acquire)).leaked()
    }

    /// Get the raw state for advanced operations
    #[inline]
    pub fn state(&self) -> &AtomicU64 {
        &self.state
    }

    /// Get PyState from current value
    #[inline]
    pub fn py_state(&self) -> PyState {
        PyState::from_raw(self.state.load(Ordering::SeqCst))
    }

    /// Mark as destructed. Returns true if successful.
    #[inline]
    pub fn mark_destructed(&self) -> bool {
        let mut old = PyState::from_raw(self.state.load(Ordering::SeqCst));
        loop {
            if old.destructed() || old.leaked() {
                return false;
            }
            if old.strong() > 0 {
                return false;
            }
            let new_state = old.with_destructed(true);
            match self.state.compare_exchange(
                old.as_raw(),
                new_state.as_raw(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(curr) => old = PyState::from_raw(curr),
            }
        }
    }
}
