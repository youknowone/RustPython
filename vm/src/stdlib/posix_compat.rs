#[pymodule(name = "posix")]
mod posix {
    use crate::builtins::PyDictRef;

    #[pyfunction]
    pub(super) fn access(_path: PyStrRef, _mode: u8, vm: &VirtualMachine) -> PyResult<bool> {
        os_unimpl("os.access", vm)
    }

    pub const SYMLINK_DIR_FD: bool = false;

    #[derive(FromArgs)]
    #[allow(unused)]
    pub(super) struct SimlinkArgs {
        #[pyarg(any)]
        src: PyPathLike,
        #[pyarg(any)]
        dst: PyPathLike,
        #[pyarg(flatten)]
        _target_is_directory: TargetIsDirectory,
        #[pyarg(flatten)]
        _dir_fd: DirFd<{ SYMLINK_DIR_FD as usize }>,
    }

    #[pyfunction]
    pub(super) fn symlink(_args: SimlinkArgs, vm: &VirtualMachine) -> PyResult<()> {
        os_unimpl("os.symlink", vm)
    }

    #[pyattr]
    fn environ(vm: &VirtualMachine) -> PyDictRef {
        let environ = vm.ctx.new_dict();
        use ffi_ext::OsStringExt;
        for (key, value) in env::vars_os() {
            environ
                .set_item(
                    vm.ctx.new_bytes(key.into_vec()),
                    vm.ctx.new_bytes(value.into_vec()),
                    vm,
                )
                .unwrap();
        }

        environ
    }

    pub(super) fn support_funcs() -> Vec<SupportFunc> {
        Vec::new()
    }
}
