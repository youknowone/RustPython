//! Reference counting implementation based on EBR (Epoch-Based Reclamation).
//!
//! This module provides a RefCount type that is compatible with EBR's memory reclamation
//! system while maintaining the original API for backward compatibility.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

// Re-export EBR types
pub use crate::ebr::{Guard, HIGH_TAG_WIDTH, cs, global_epoch};

// ============================================================================
// Constants from circ::utils - now defined locally
// ============================================================================

pub const EPOCH_WIDTH: u32 = HIGH_TAG_WIDTH;
pub const EPOCH_MASK_HEIGHT: u32 = u64::BITS - EPOCH_WIDTH;
pub const EPOCH: u64 = ((1 << EPOCH_WIDTH) - 1) << EPOCH_MASK_HEIGHT;
pub const DESTRUCTED: u64 = 1 << (EPOCH_MASK_HEIGHT - 1);
pub const WEAKED: u64 = 1 << (EPOCH_MASK_HEIGHT - 2);
pub const TOTAL_COUNT_WIDTH: u32 = u64::BITS - EPOCH_WIDTH - 2;
pub const WEAK_WIDTH: u32 = TOTAL_COUNT_WIDTH / 2;
pub const STRONG_WIDTH: u32 = TOTAL_COUNT_WIDTH - WEAK_WIDTH;
pub const STRONG: u64 = (1 << STRONG_WIDTH) - 1;
pub const WEAK: u64 = ((1 << WEAK_WIDTH) - 1) << STRONG_WIDTH;
pub const COUNT: u64 = 1;
pub const WEAK_COUNT: u64 = 1 << STRONG_WIDTH;

/// LEAKED bit for interned objects (never deallocated)
/// Position: just after WEAKED bit
pub const LEAKED: u64 = 1 << (EPOCH_MASK_HEIGHT - 3);

/// State wraps reference count + flags in a single 64-bit word
#[derive(Clone, Copy)]
pub struct State {
    inner: u64,
}

impl State {
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

/// Modular arithmetic for epoch comparisons
pub struct Modular<const WIDTH: u32> {
    max: isize,
}

impl<const WIDTH: u32> Modular<WIDTH> {
    /// Creates a modular space where `max` is the maximum.
    pub fn new(max: isize) -> Self {
        Self { max }
    }

    // Sends a number to a modular space.
    pub fn trans(&self, val: isize) -> isize {
        debug_assert!(val <= self.max);
        (val - (self.max + 1)) % (1 << WIDTH)
    }

    // Receives a number from a modular space.
    pub fn inver(&self, val: isize) -> isize {
        (val + (self.max + 1)) % (1 << WIDTH)
    }

    pub fn max(&self, nums: &[isize]) -> isize {
        self.inver(nums.iter().fold(isize::MIN, |acc, val| {
            acc.max(self.trans(val % (1 << WIDTH)))
        }))
    }

    // Checks if `a` is less than or equal to `b` in the modular space.
    pub fn le(&self, a: isize, b: isize) -> bool {
        self.trans(a) <= self.trans(b)
    }
}

/// PyState extends State with LEAKED support for RustPython
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

/// Reference count using state layout with LEAKED support.
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

// ============================================================================
// Deferred Drop Infrastructure
// ============================================================================
//
// This mechanism allows untrack_object() calls to be deferred until after
// the GC collection phase completes, preventing deadlocks that occur when
// pop_edges() triggers object destruction while holding the tracked_objects lock.

thread_local! {
    /// Flag indicating if we're inside a deferred drop context.
    /// When true, drop operations should defer untrack calls.
    static IN_DEFERRED_CONTEXT: Cell<bool> = const { Cell::new(false) };

    /// Queue of deferred untrack operations.
    /// No Send bound needed - this is thread-local and only accessed from the same thread.
    static DEFERRED_QUEUE: RefCell<Vec<Box<dyn FnOnce()>>> = const { RefCell::new(Vec::new()) };
}

/// Execute a function within a deferred drop context.
/// Any calls to `try_defer_drop` within this context will be queued
/// and executed when the context exits.
#[inline]
pub fn with_deferred_drops<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    IN_DEFERRED_CONTEXT.with(|in_ctx| {
        let was_in_context = in_ctx.get();
        in_ctx.set(true);
        let result = f();
        in_ctx.set(was_in_context);

        // Only flush if we're the outermost context
        if !was_in_context {
            flush_deferred_drops();
        }

        result
    })
}

/// Try to defer a drop-related operation.
/// If inside a deferred context, the operation is queued.
/// Otherwise, it executes immediately.
///
/// Note: No `Send` bound - this is thread-local and runs on the same thread.
#[inline]
pub fn try_defer_drop<F>(f: F)
where
    F: FnOnce() + 'static,
{
    let should_defer = IN_DEFERRED_CONTEXT.with(|in_ctx| in_ctx.get());

    if should_defer {
        DEFERRED_QUEUE.with(|q| {
            q.borrow_mut().push(Box::new(f));
        });
    } else {
        f();
    }
}

/// Flush all deferred drop operations.
/// This is automatically called when exiting a deferred context.
#[inline]
pub fn flush_deferred_drops() {
    DEFERRED_QUEUE.with(|q| {
        // Take all queued operations
        let ops: Vec<_> = q.borrow_mut().drain(..).collect();
        // Execute them outside the borrow
        for op in ops {
            op();
        }
    });
}

/// Defer a closure execution using EBR until all pinned threads unpin.
///
/// This function queues a closure to be executed only after all currently
/// pinned threads (those in EBR critical sections) have exited their
/// critical sections. This is the 3-epoch guarantee of EBR.
///
/// # Safety
///
/// - The closure must not hold references to the stack
/// - The closure must be `Send` (may execute on a different thread)
/// - Should only be called within an EBR critical section (with a valid Guard)
#[inline]
pub unsafe fn defer_destruction<F>(guard: &Guard, f: F)
where
    F: FnOnce() + Send + 'static,
{
    // SAFETY: Caller guarantees the closure is safe to defer
    unsafe { guard.defer_unchecked(f) };
}
