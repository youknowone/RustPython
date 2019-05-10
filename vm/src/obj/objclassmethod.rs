use super::objtype::PyClassRef;
use crate::pyobject::{PyClassImpl, PyContext, PyObjectRef, PyRef, PyResult, PyValue};
use crate::vm::VirtualMachine;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyClassMethod {
    pub callable: PyObjectRef,
}
pub type PyClassMethodRef = PyRef<PyClassMethod>;

impl PyValue for PyClassMethod {
    const HAVE_DICT: bool = true;

    fn class(vm: &VirtualMachine) -> PyClassRef {
        vm.ctx.classmethod_type()
    }
}

#[pyimpl]
impl PyClassMethod {
    #[pymethod(name = "__new__")]
    fn new(
        cls: PyClassRef,
        callable: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyClassMethodRef> {
        PyClassMethod {
            callable: callable.clone(),
        }
        .into_ref_with_type(vm, cls)
    }

    #[pymethod(name = "__get__")]
    fn get(&self, _inst: PyObjectRef, owner: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        Ok(vm
            .ctx
            .new_bound_method(self.callable.clone(), owner.clone()))
    }
}

pub fn init(context: &PyContext) {
    PyClassMethod::extend_class(context, &context.classmethod_type);
}
