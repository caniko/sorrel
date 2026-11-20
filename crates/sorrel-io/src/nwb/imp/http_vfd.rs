use crate::http_range::Object;
use anyhow::{Context, Result};
use hdf5_metno as h5;
use hdf5_metno_sys::h5i::hid_t;
use std::{
    ffi::{c_char, c_int, c_void, CString},
    panic::catch_unwind,
    sync::Arc,
};

unsafe extern "C" {
    fn sorrel_range_open(object: *const c_void, size: u64) -> hid_t;
    fn sorrel_range_error(message: *const c_char);
}

pub(super) fn open(object: &Arc<Object>) -> Result<h5::File> {
    h5::sync::sync(|| {
        // The shim retains a reference for each property list and file it owns.
        let id = unsafe { sorrel_range_open(Arc::as_ptr(object).cast(), object.size) };
        if id < 0 {
            return Err(anyhow::anyhow!("{}", h5::Error::query()?))
                .with_context(|| format!("open remote NWB {}", object.label));
        }
        Ok(unsafe { h5::from_id(id) }?)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sorrel_range_retain(object: *const Object) {
    // Only called with a live Arc allocation.
    unsafe { Arc::increment_strong_count(object) };
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sorrel_range_release(object: *const Object) {
    // Each call balances a retain. Don't unwind through HDF5 on drop.
    let _ = catch_unwind(|| unsafe { Arc::decrement_strong_count(object) });
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sorrel_range_read(
    object: *const Object,
    addr: u64,
    size: usize,
    buffer: *mut u8,
) -> c_int {
    match catch_unwind(|| {
        if size == 0 {
            return 0;
        }
        // HDF5 supplies a writable, non-null buffer of `size` bytes. Use
        // MaybeUninit because HDF5 needn't have initialized it beforehand.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buffer.cast(), size) };
        unsafe { Arc::increment_strong_count(object) };
        let object = unsafe { Arc::from_raw(object) };
        match object.read_uninit(addr, bytes, true) {
            Ok(()) => 0,
            Err(error) => {
                if let Ok(message) = CString::new(format!("{error:#}")) {
                    unsafe { sorrel_range_error(message.as_ptr()) };
                }
                -1
            }
        }
    }) {
        Ok(status) => status,
        Err(_) => {
            unsafe { sorrel_range_error(c"HTTP range reader panicked".as_ptr()) };
            -1
        }
    }
}
