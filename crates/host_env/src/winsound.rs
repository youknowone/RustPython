// spell-checker:ignore pszSound fdwSound winmm

use std::io;

#[link(name = "winmm")]
unsafe extern "system" {
    fn PlaySoundW(pszSound: *const u16, hmod: isize, fdwSound: u32) -> i32;
}

unsafe extern "system" {
    fn Beep(dwFreq: u32, dwDuration: u32) -> i32;
    fn MessageBeep(uType: u32) -> i32;
}

/// `PlaySoundW` flags from `mmsystem.h`. `SND_SYNC` is the absence of
/// `SND_ASYNC`, not a bit of its own.
pub const SND_SYNC: u32 = 0x0000;
pub const SND_ASYNC: u32 = 0x0001;
pub const SND_NODEFAULT: u32 = 0x0002;
pub const SND_MEMORY: u32 = 0x0004;
pub const SND_LOOP: u32 = 0x0008;
pub const SND_NOSTOP: u32 = 0x0010;
pub const SND_PURGE: u32 = 0x0040;
pub const SND_APPLICATION: u32 = 0x0080;
pub const SND_NOWAIT: u32 = 0x0000_2000;
pub const SND_ALIAS: u32 = 0x0001_0000;
pub const SND_FILENAME: u32 = 0x0002_0000;
pub const SND_SENTRY: u32 = 0x0008_0000;
pub const SND_SYSTEM: u32 = 0x0020_0000;

/// `MessageBeep` sound ids from `winuser.h`. Several names are one sound.
pub const MB_OK: u32 = 0x0000_0000;
pub const MB_ICONHAND: u32 = 0x0000_0010;
pub const MB_ICONQUESTION: u32 = 0x0000_0020;
pub const MB_ICONEXCLAMATION: u32 = 0x0000_0030;
pub const MB_ICONASTERISK: u32 = 0x0000_0040;
pub const MB_ICONERROR: u32 = MB_ICONHAND;
pub const MB_ICONSTOP: u32 = MB_ICONHAND;
pub const MB_ICONINFORMATION: u32 = MB_ICONASTERISK;
pub const MB_ICONWARNING: u32 = MB_ICONEXCLAMATION;

/// Source for a `PlaySound` call.
pub enum PlaySoundSource<'a> {
    /// Stop currently playing sound (NULL `pszSound`).
    Stop,
    /// Play sound data from memory; pass with `SND_MEMORY` set in `flags`.
    Memory(&'a [u8]),
    /// Play sound by filename or system alias.
    Name(&'a widestring::WideCStr),
}

/// Returns `Ok(())` when `PlaySoundW` returns non-zero, an error otherwise.
///
/// Rejects `Memory(_)` together with `SND_ASYNC` because async playback
/// requires the buffer to outlive the call; combining them with a borrowed
/// slice would let WinMM read freed memory.
pub fn play_sound(source: PlaySoundSource<'_>, flags: u32) -> Result<(), PlaySoundError> {
    if matches!(source, PlaySoundSource::Memory(_)) && flags & SND_ASYNC != 0 {
        return Err(PlaySoundError::MemoryAsyncRejected);
    }
    // `SND_MEMORY` requires a `Memory(_)` source; an empty pointer would
    // dereference garbage.
    if !matches!(source, PlaySoundSource::Memory(_)) && flags & SND_MEMORY != 0 {
        return Err(PlaySoundError::MemoryFlagWithoutBuffer);
    }
    let ptr: *const u16 = match source {
        PlaySoundSource::Stop => core::ptr::null(),
        PlaySoundSource::Memory(buf) => buf.as_ptr().cast(),
        PlaySoundSource::Name(s) => s.as_ptr(),
    };
    let ok = unsafe { PlaySoundW(ptr, 0, flags) };
    if ok == 0 {
        Err(PlaySoundError::CallFailed)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PlaySoundError {
    /// `PlaySoundW` returned 0; there is no documented errno for this path.
    CallFailed,
    /// `Memory(_)` source cannot be combined with `SND_ASYNC` in the safe API.
    MemoryAsyncRejected,
    /// `SND_MEMORY` set in `flags` but no `Memory(_)` buffer supplied.
    MemoryFlagWithoutBuffer,
}

/// `Beep(freq, duration)`. `false` on failure.
#[must_use]
pub fn beep(frequency: u32, duration_ms: u32) -> bool {
    unsafe { Beep(frequency, duration_ms) != 0 }
}

/// `MessageBeep(type)`. On failure returns `Err` populated from `GetLastError`.
pub fn message_beep(beep_type: u32) -> io::Result<()> {
    let ok = unsafe { MessageBeep(beep_type) };
    if ok == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}
