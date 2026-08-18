//! Raw locks that let a thread leave its interpreter before it blocks.
//!
//! Stopping the world means waiting for every running thread to reach a
//! safepoint. A thread blocked on a lock reaches none, so if the thread
//! holding that lock has already been stopped, the two wait on each other
//! forever. The holder is not the one who can avoid this — a lock is held
//! across a blocking call precisely because that is what the call needs — so
//! the waiter gives up its interpreter for the duration of the wait instead,
//! which is what a blocking call does anyway.
//!
//! Only the contended path pays for this: an acquire that takes the lock on
//! the first try is the same atomic exchange it was before, and never reaches
//! the hook at all. The hook is installed by whoever knows how
//! to detach a thread ([`set_blocking_wait_hook`]); until then, and on any
//! thread that is not running an interpreter, a blocked acquire just blocks.

use core::cell::Cell;
use lock_api::{
    RawMutex as RawMutexTrait, RawRwLock as RawRwLockTrait, RawRwLockDowngrade,
    RawRwLockRecursive as RawRwLockRecursiveTrait, RawRwLockUpgrade as RawRwLockUpgradeTrait,
    RawRwLockUpgradeDowngrade,
};
use std::sync::OnceLock;

/// Runs `wait` with the calling thread detached from its interpreter.
pub type BlockingWaitHook = fn(wait: &dyn Fn());

static BLOCKING_WAIT: OnceLock<BlockingWaitHook> = OnceLock::new();

/// Install the hook that detaches a thread around a blocked lock acquire.
///
/// Later calls are ignored, so every interpreter in a process can call this
/// during its own initialization.
pub fn set_blocking_wait_hook(hook: BlockingWaitHook) {
    let _ = BLOCKING_WAIT.set(hook);
}

std::thread_local! {
    /// Set while this thread is inside the hook, so that a lock taken by the
    /// hook itself — or by anything detaching and re-attaching runs — waits
    /// plainly instead of recursing back into it.
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// Clears [`IN_HOOK`] even if the hook unwinds.
struct HookGuard;

impl Drop for HookGuard {
    fn drop(&mut self) {
        let _ = IN_HOOK.try_with(|in_hook| in_hook.set(false));
    }
}

/// Block on `wait`, detached from this thread's interpreter if there is one.
#[cold]
#[inline(never)]
fn wait_detached(wait: impl Fn()) {
    let Some(hook) = BLOCKING_WAIT.get() else {
        wait();
        return;
    };
    // `try_with` fails once the thread's locals are being destroyed, which is
    // also a point at which there is no interpreter left to detach from.
    let entered = IN_HOOK
        .try_with(|in_hook| !in_hook.replace(true))
        .unwrap_or(false);
    if !entered {
        wait();
        return;
    }
    let _guard = HookGuard;
    hook(&wait);
}

/// A mutex whose blocking acquire detaches first. `parking_lot::RawMutex`
/// otherwise.
#[repr(transparent)]
pub struct RawMutex(parking_lot::RawMutex);

// SAFETY: every method forwards to `parking_lot::RawMutex`, which upholds the
// contract; `lock` only adds a wait that ends with the same lock acquired.
unsafe impl RawMutexTrait for RawMutex {
    #[allow(
        clippy::declare_interior_mutable_const,
        reason = "raw lock initializer, as in the type it wraps"
    )]
    const INIT: Self = Self(<parking_lot::RawMutex as RawMutexTrait>::INIT);

    type GuardMarker = <parking_lot::RawMutex as RawMutexTrait>::GuardMarker;

    #[inline]
    fn lock(&self) {
        if !self.0.try_lock() {
            wait_detached(|| self.0.lock());
        }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        self.0.try_lock()
    }

    #[inline]
    unsafe fn unlock(&self) {
        unsafe { self.0.unlock() }
    }

    #[inline]
    fn is_locked(&self) -> bool {
        self.0.is_locked()
    }
}

/// A reader-writer lock whose blocking acquires detach first.
/// `parking_lot::RawRwLock` otherwise.
#[repr(transparent)]
pub struct RawRwLock(parking_lot::RawRwLock);

// SAFETY: every method forwards to `parking_lot::RawRwLock`, which upholds the
// contract; the blocking acquires only add a wait that ends with the same lock
// acquired.
unsafe impl RawRwLockTrait for RawRwLock {
    #[allow(
        clippy::declare_interior_mutable_const,
        reason = "raw lock initializer, as in the type it wraps"
    )]
    const INIT: Self = Self(<parking_lot::RawRwLock as RawRwLockTrait>::INIT);

    type GuardMarker = <parking_lot::RawRwLock as RawRwLockTrait>::GuardMarker;

    #[inline]
    fn lock_shared(&self) {
        if !self.0.try_lock_shared() {
            wait_detached(|| self.0.lock_shared());
        }
    }

    #[inline]
    fn try_lock_shared(&self) -> bool {
        self.0.try_lock_shared()
    }

    #[inline]
    unsafe fn unlock_shared(&self) {
        unsafe { self.0.unlock_shared() }
    }

    #[inline]
    fn lock_exclusive(&self) {
        if !self.0.try_lock_exclusive() {
            wait_detached(|| self.0.lock_exclusive());
        }
    }

    #[inline]
    fn try_lock_exclusive(&self) -> bool {
        self.0.try_lock_exclusive()
    }

    #[inline]
    unsafe fn unlock_exclusive(&self) {
        unsafe { self.0.unlock_exclusive() }
    }

    #[inline]
    fn is_locked(&self) -> bool {
        self.0.is_locked()
    }

    #[inline]
    fn is_locked_exclusive(&self) -> bool {
        self.0.is_locked_exclusive()
    }
}

// SAFETY: forwards to `parking_lot::RawRwLock`.
unsafe impl RawRwLockDowngrade for RawRwLock {
    #[inline]
    unsafe fn downgrade(&self) {
        unsafe { self.0.downgrade() }
    }
}

// SAFETY: forwards to `parking_lot::RawRwLock`; the blocking acquires only add
// a wait that ends with the same lock acquired.
unsafe impl RawRwLockUpgradeTrait for RawRwLock {
    #[inline]
    fn lock_upgradable(&self) {
        if !self.0.try_lock_upgradable() {
            wait_detached(|| self.0.lock_upgradable());
        }
    }

    #[inline]
    fn try_lock_upgradable(&self) -> bool {
        self.0.try_lock_upgradable()
    }

    #[inline]
    unsafe fn unlock_upgradable(&self) {
        unsafe { self.0.unlock_upgradable() }
    }

    #[inline]
    unsafe fn upgrade(&self) {
        // SAFETY: the caller holds the upgradable lock, as `upgrade` requires,
        // and it stays held for both the failed attempt and the wait.
        unsafe {
            if !self.0.try_upgrade() {
                wait_detached(|| self.0.upgrade());
            }
        }
    }

    #[inline]
    unsafe fn try_upgrade(&self) -> bool {
        unsafe { self.0.try_upgrade() }
    }
}

// SAFETY: forwards to `parking_lot::RawRwLock`.
unsafe impl RawRwLockUpgradeDowngrade for RawRwLock {
    #[inline]
    unsafe fn downgrade_upgradable(&self) {
        unsafe { self.0.downgrade_upgradable() }
    }

    #[inline]
    unsafe fn downgrade_to_upgradable(&self) {
        unsafe { self.0.downgrade_to_upgradable() }
    }
}

// SAFETY: forwards to `parking_lot::RawRwLock`; the blocking acquire only adds
// a wait that ends with the same lock acquired.
unsafe impl RawRwLockRecursiveTrait for RawRwLock {
    #[inline]
    fn lock_shared_recursive(&self) {
        if !self.0.try_lock_shared_recursive() {
            wait_detached(|| self.0.lock_shared_recursive());
        }
    }

    #[inline]
    fn try_lock_shared_recursive(&self) -> bool {
        self.0.try_lock_shared_recursive()
    }
}
