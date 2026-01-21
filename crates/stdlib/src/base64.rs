// Base64 encoding module

pub(crate) use _base64::make_module;

const PAD_BYTE: u8 = b'=';
const ENCODE_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[inline]
fn encoded_output_len(input_len: usize) -> Option<usize> {
    input_len
        .checked_add(2)
        .map(|n| n / 3)
        .and_then(|blocks| blocks.checked_mul(4))
}

#[inline]
fn encode_into(input: &[u8], output: &mut [u8]) -> usize {
    let mut src_index = 0;
    let mut dst_index = 0;
    let len = input.len();

    // Process full 3-byte chunks
    while src_index + 3 <= len {
        let chunk = (u32::from(input[src_index]) << 16)
            | (u32::from(input[src_index + 1]) << 8)
            | u32::from(input[src_index + 2]);
        output[dst_index] = ENCODE_TABLE[((chunk >> 18) & 0x3f) as usize];
        output[dst_index + 1] = ENCODE_TABLE[((chunk >> 12) & 0x3f) as usize];
        output[dst_index + 2] = ENCODE_TABLE[((chunk >> 6) & 0x3f) as usize];
        output[dst_index + 3] = ENCODE_TABLE[(chunk & 0x3f) as usize];
        src_index += 3;
        dst_index += 4;
    }

    // Process remaining bytes (1 or 2 bytes)
    match len - src_index {
        0 => {}
        1 => {
            let chunk = u32::from(input[src_index]) << 16;
            output[dst_index] = ENCODE_TABLE[((chunk >> 18) & 0x3f) as usize];
            output[dst_index + 1] = ENCODE_TABLE[((chunk >> 12) & 0x3f) as usize];
            output[dst_index + 2] = PAD_BYTE;
            output[dst_index + 3] = PAD_BYTE;
            dst_index += 4;
        }
        2 => {
            let chunk =
                (u32::from(input[src_index]) << 16) | (u32::from(input[src_index + 1]) << 8);
            output[dst_index] = ENCODE_TABLE[((chunk >> 18) & 0x3f) as usize];
            output[dst_index + 1] = ENCODE_TABLE[((chunk >> 12) & 0x3f) as usize];
            output[dst_index + 2] = ENCODE_TABLE[((chunk >> 6) & 0x3f) as usize];
            output[dst_index + 3] = PAD_BYTE;
            dst_index += 4;
        }
        _ => unreachable!("len - src_index cannot exceed 2"),
    }

    dst_index
}

#[pymodule(name = "_base64")]
mod _base64 {
    use rustpython_vm::builtins::PyBytes;

    use crate::vm::{PyResult, VirtualMachine, function::ArgBytesLike};

    // argument is parsed by #[pyfunction]. It replaces clinic.
    #[pyfunction]
    fn standard_b64encode(data: ArgBytesLike, vm: &VirtualMachine) -> PyResult<PyBytes> {
        data.with_ref(|input| {
            let input_len = input.len();

            // input len can't be negative in RustPython

            let Some(output_len) = super::encoded_output_len(input_len) else {
                return Err(vm.new_memory_error("output length overflow".to_owned()));
            };

            if output_len > isize::MAX as usize {
                return Err(vm.new_memory_error("output too large".to_owned()));
            }

            let mut output = vec![0u8; output_len];
            let written = super::encode_into(input, &mut output);
            debug_assert_eq!(written, output_len);

            Ok(PyBytes::from(output))
        })
    }
}
