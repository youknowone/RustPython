pub(crate) use sysconfig::make_module;

#[pymodule(name = "_sysconfig")]
pub(crate) mod sysconfig {
    use crate::{VirtualMachine, builtins::PyDictRef, convert::ToPyObject};

    #[pyfunction]
    fn config_vars(vm: &VirtualMachine) -> PyDictRef {
        let vars = vm.ctx.new_dict();

        // RustPython doesn't support native extensions, so EXT_SUFFIX is None
        // This causes pip's _generic_abi() to return an empty list (correct behavior)
        vars.set_item("EXT_SUFFIX", vm.ctx.none(), vm).unwrap();
        vars.set_item("SOABI", vm.ctx.none(), vm).unwrap();

        vars.set_item("Py_GIL_DISABLED", true.to_pyobject(vm), vm)
            .unwrap();
        vars.set_item("Py_DEBUG", false.to_pyobject(vm), vm)
            .unwrap();

        vars
    }
}
