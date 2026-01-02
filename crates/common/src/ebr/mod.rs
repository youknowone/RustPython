//! Epoch-based memory reclamation (EBR).
//!
//! This module provides safe memory reclamation for lock-free data structures
//! using epoch-based reclamation. It is based on crossbeam-epoch.
//!
//! # Overview
//!
//! When an element gets removed from a concurrent collection, it is inserted into
//! a pile of garbage and marked with the current epoch. Every time a thread accesses
//! a collection, it checks the current epoch, attempts to increment it, and destructs
//! some garbage that became so old that no thread can be referencing it anymore.
//!
//! # Pinning
//!
//! Before accessing shared data, a participant must be pinned using [`cs`]. This
//! returns a [`Guard`] that keeps the thread in a critical section until dropped.

mod collector;
mod default;
mod deferred;
mod epoch;
mod guard;
pub mod internal;
mod pointers;
mod sync;

pub use default::*;
pub use epoch::*;
pub use guard::*;
pub use pointers::*;
