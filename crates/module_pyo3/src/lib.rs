//! RustPython to CPython bridge via PyO3
//!
//! This crate provides interoperability between RustPython and CPython,
//! allowing RustPython code to execute functions in the CPython runtime.
//!
//! # Background
//!
//! RustPython does not implement all CPython C extension modules.
//! This crate enables calling into the real CPython runtime for functionality
//! that is not yet available in RustPython.
//!
//! # Architecture
//!
//! Communication between RustPython and CPython uses PyO3 for in-process calls.
//! Data is serialized using Python's `pickle` protocol:
//!
//! ```text
//! RustPython                         CPython
//!     │                                  │
//!     │  pickle.dumps(args, kwargs)      │
//!     │ ──────────────────────────────►  │
//!     │                                  │  exec(source)
//!     │                                  │  result = func(*args, **kwargs)
//!     │  pickle.dumps(result)            │
//!     │ ◄──────────────────────────────  │
//!     │                                  │
//!     │  pickle.loads(result)            │
//! ```
//!
//! # Limitations
//!
//! - **File-based functions only**: Functions defined in REPL or via `exec()` will fail
//!   (`inspect.getsource()` requires source file access)
//! - **Picklable data only**: Cannot pass functions, classes, file handles, etc.
//! - **Performance overhead**: pickle serialization + CPython GIL acquisition
//! - **CPython required**: System must have CPython installed (linked via PyO3)

#[macro_use]
extern crate rustpython_derive;

use rustpython_vm::{PyRef, VirtualMachine, builtins::PyModule};

/// Create the pyo3 module
pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    _pyo3::make_module(vm)
}

/// Borrow a module from CPython and register it in sys.modules.
///
/// This allows RustPython to use CPython's implementation of a module
/// (e.g., _ctypes) instead of implementing it in Rust.
pub fn borrow_module(
    name: &str,
    vm: &VirtualMachine,
) -> Result<(), rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>> {
    vm.import("pyo3", 0)?;

    let module = _pyo3::import_module_impl(name, vm)?;

    // Register in sys.modules
    let sys_modules = vm.sys_module.get_attr("modules", vm)?;
    sys_modules.set_item(name, module, vm)?;

    Ok(())
}

#[pymodule]
mod _pyo3 {
    use crossbeam_utils::atomic::AtomicCell;
    use pyo3::PyErr;
    use pyo3::prelude::PyAnyMethods;
    use pyo3::types::PyBytes as Pyo3Bytes;
    use pyo3::types::PyBytesMethods;
    use pyo3::types::PyCFunction;
    use pyo3::types::PyDictMethods;
    use pyo3::types::PyTupleMethods;
    use rustpython_vm::{
        AsObject, Py, PyObject, PyObjectRef, PyPayload, PyRef, PyResult, TryFromBorrowedObject,
        VirtualMachine,
        builtins::{
            PyBaseExceptionRef, PyBytes as RustPyBytes, PyBytesRef, PyDict, PyStr, PyStrRef,
            PyType, PyTypeRef,
        },
        function::{FuncArgs, PyArithmeticValue, PyComparisonValue, PySetterValue},
        protocol::{PyBuffer, PyIterReturn, PyMappingMethods, PyNumberMethods, PySequenceMethods},
        types::{
            AsMapping, AsNumber, AsSequence, Callable, Comparable, Constructor, GetAttr,
            GetDescriptor, Iterable, PyComparisonOp, Representable, SetAttr,
        },
    };
    use std::sync::Arc;

    /// Global storage for buffer guards, keyed by memoryview pointer.
    /// Keeps capsules (which hold BufferGuards) alive while CPython uses the shared memory.
    /// Note: This is a simplified implementation that leaks memory.
    static BUFFER_GUARDS: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<isize, pyo3::Py<pyo3::PyAny>>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    /// Cache for Pyo3Type wrappers. Maps CPython type pointer → RustPython type.
    /// Ensures type identity: same CPython type always returns same RustPython wrapper.
    static TYPE_CACHE: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<isize, PyTypeRef>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    /// Identifier for CPython builtin types that should map to RustPython native types.
    #[derive(Clone, Copy, Debug)]
    enum BuiltinTypeId {
        Type,
        Object,
        Int,
        Float,
        Complex,
        Bool,
        Str,
        Bytes,
        ByteArray,
        List,
        Tuple,
        Set,
        FrozenSet,
        Dict,
        Slice,
        MemoryView,
        NoneType,
        NotImplementedType,
        EllipsisType,
        Range,
        Property,
        StaticMethod,
        ClassMethod,
        Super,
        Filter,
        Map,
        Zip,
        Enumerate,
        BaseException,
        Exception,
    }

    /// Cache for CPython builtin type pointers → BuiltinTypeId.
    /// Built lazily on first access, maps CPython type pointers to identifiers.
    static BUILTIN_TYPE_MAP: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<isize, BuiltinTypeId>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    /// Initialize the builtin type map with CPython type pointers.
    /// Called once per Python interpreter to build the mapping.
    fn init_builtin_type_map(py: pyo3::Python<'_>) {
        use pyo3::types::{
            PyBool, PyByteArray, PyBytes, PyComplex, PyDict, PyFloat, PyFrozenSet, PyInt, PyList,
            PyMemoryView, PySet, PySlice, PyString, PyTuple, PyType,
        };

        let mut map = BUILTIN_TYPE_MAP.lock().unwrap();
        if !map.is_empty() {
            return; // Already initialized
        }

        // Helper macro to insert type mapping
        macro_rules! insert_type {
            ($py_type:ty, $id:expr) => {
                map.insert(py.get_type::<$py_type>().as_ptr() as isize, $id);
            };
        }

        // Core types
        insert_type!(PyType, BuiltinTypeId::Type);
        insert_type!(pyo3::types::PyAny, BuiltinTypeId::Object);

        // Numeric types
        insert_type!(PyInt, BuiltinTypeId::Int);
        insert_type!(PyFloat, BuiltinTypeId::Float);
        insert_type!(PyComplex, BuiltinTypeId::Complex);
        insert_type!(PyBool, BuiltinTypeId::Bool);

        // Sequence types
        insert_type!(PyString, BuiltinTypeId::Str);
        insert_type!(PyBytes, BuiltinTypeId::Bytes);
        insert_type!(PyByteArray, BuiltinTypeId::ByteArray);
        insert_type!(PyList, BuiltinTypeId::List);
        insert_type!(PyTuple, BuiltinTypeId::Tuple);

        // Set types
        insert_type!(PySet, BuiltinTypeId::Set);
        insert_type!(PyFrozenSet, BuiltinTypeId::FrozenSet);

        // Mapping types
        insert_type!(PyDict, BuiltinTypeId::Dict);

        // Other types
        insert_type!(PySlice, BuiltinTypeId::Slice);
        insert_type!(PyMemoryView, BuiltinTypeId::MemoryView);

        // Singleton types
        map.insert(
            py.None().bind(py).get_type().as_ptr() as isize,
            BuiltinTypeId::NoneType,
        );
        map.insert(
            py.NotImplemented().bind(py).get_type().as_ptr() as isize,
            BuiltinTypeId::NotImplementedType,
        );

        // Ellipsis type
        if let Ok(ellipsis) = py.eval(pyo3::ffi::c_str!("..."), None, None) {
            map.insert(
                ellipsis.get_type().as_ptr() as isize,
                BuiltinTypeId::EllipsisType,
            );
        }

        // Get additional builtins from the builtins module
        if let Ok(builtins) = py.import("builtins") {
            macro_rules! insert_builtin {
                ($name:literal, $id:expr) => {
                    if let Ok(typ) = builtins.getattr($name) {
                        map.insert(typ.as_ptr() as isize, $id);
                    }
                };
            }

            insert_builtin!("range", BuiltinTypeId::Range);
            insert_builtin!("property", BuiltinTypeId::Property);
            insert_builtin!("staticmethod", BuiltinTypeId::StaticMethod);
            insert_builtin!("classmethod", BuiltinTypeId::ClassMethod);
            insert_builtin!("super", BuiltinTypeId::Super);
            insert_builtin!("filter", BuiltinTypeId::Filter);
            insert_builtin!("map", BuiltinTypeId::Map);
            insert_builtin!("zip", BuiltinTypeId::Zip);
            insert_builtin!("enumerate", BuiltinTypeId::Enumerate);
            insert_builtin!("BaseException", BuiltinTypeId::BaseException);
            insert_builtin!("Exception", BuiltinTypeId::Exception);
        }
    }

    /// Resolve a BuiltinTypeId to the corresponding RustPython native type.
    fn resolve_builtin_type_id(id: BuiltinTypeId, vm: &VirtualMachine) -> PyTypeRef {
        match id {
            BuiltinTypeId::Type => vm.ctx.types.type_type.to_owned(),
            BuiltinTypeId::Object => vm.ctx.types.object_type.to_owned(),
            BuiltinTypeId::Int => vm.ctx.types.int_type.to_owned(),
            BuiltinTypeId::Float => vm.ctx.types.float_type.to_owned(),
            BuiltinTypeId::Complex => vm.ctx.types.complex_type.to_owned(),
            BuiltinTypeId::Bool => vm.ctx.types.bool_type.to_owned(),
            BuiltinTypeId::Str => vm.ctx.types.str_type.to_owned(),
            BuiltinTypeId::Bytes => vm.ctx.types.bytes_type.to_owned(),
            BuiltinTypeId::ByteArray => vm.ctx.types.bytearray_type.to_owned(),
            BuiltinTypeId::List => vm.ctx.types.list_type.to_owned(),
            BuiltinTypeId::Tuple => vm.ctx.types.tuple_type.to_owned(),
            BuiltinTypeId::Set => vm.ctx.types.set_type.to_owned(),
            BuiltinTypeId::FrozenSet => vm.ctx.types.frozenset_type.to_owned(),
            BuiltinTypeId::Dict => vm.ctx.types.dict_type.to_owned(),
            BuiltinTypeId::Slice => vm.ctx.types.slice_type.to_owned(),
            BuiltinTypeId::MemoryView => vm.ctx.types.memoryview_type.to_owned(),
            BuiltinTypeId::NoneType => vm.ctx.types.none_type.to_owned(),
            BuiltinTypeId::NotImplementedType => vm.ctx.types.not_implemented_type.to_owned(),
            BuiltinTypeId::EllipsisType => vm.ctx.types.ellipsis_type.to_owned(),
            BuiltinTypeId::Range => vm.ctx.types.range_type.to_owned(),
            BuiltinTypeId::Property => vm.ctx.types.property_type.to_owned(),
            BuiltinTypeId::StaticMethod => vm.ctx.types.staticmethod_type.to_owned(),
            BuiltinTypeId::ClassMethod => vm.ctx.types.classmethod_type.to_owned(),
            BuiltinTypeId::Super => vm.ctx.types.super_type.to_owned(),
            BuiltinTypeId::Filter => vm.ctx.types.filter_type.to_owned(),
            BuiltinTypeId::Map => vm.ctx.types.map_type.to_owned(),
            BuiltinTypeId::Zip => vm.ctx.types.zip_type.to_owned(),
            BuiltinTypeId::Enumerate => vm.ctx.types.enumerate_type.to_owned(),
            BuiltinTypeId::BaseException => vm.ctx.exceptions.base_exception_type.to_owned(),
            BuiltinTypeId::Exception => vm.ctx.exceptions.exception_type.to_owned(),
        }
    }

    /// Convert pyo3::PyErr to RustPython exception with proper type mapping.
    /// This preserves the original exception type from CPython instead of wrapping
    /// everything as RuntimeError.
    fn pyo3_err_to_rustpython(e: pyo3::PyErr, vm: &VirtualMachine) -> PyBaseExceptionRef {
        use pyo3::types::PyTypeMethods;

        pyo3::Python::attach(|py| {
            let err_type = e.get_type(py);
            let err_msg = e.value(py).to_string();
            let type_name = err_type
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "RuntimeError".to_string());

            match type_name.as_str() {
                "TypeError" => vm.new_type_error(err_msg),
                "ValueError" => vm.new_value_error(err_msg),
                "AttributeError" => vm.new_attribute_error(err_msg),
                "KeyError" => vm.new_key_error(vm.ctx.new_str(err_msg).into()),
                "IndexError" => vm.new_index_error(err_msg),
                "NameError" => vm.new_name_error(err_msg, vm.ctx.new_str("")),
                "StopIteration" => vm.new_stop_iteration(None),
                "OSError" | "IOError" => vm.new_os_error(err_msg),
                "NotImplementedError" => vm.new_not_implemented_error(err_msg),
                "OverflowError" => vm.new_overflow_error(err_msg),
                "ZeroDivisionError" => vm.new_zero_division_error(err_msg),
                "RecursionError" => vm.new_recursion_error(err_msg),
                "MemoryError" => vm.new_memory_error(err_msg),
                _ => vm.new_runtime_error(err_msg),
            }
        })
    }

    /// Wrapper class for executing functions in CPython.
    /// Used as a decorator: @cpython.wraps
    #[pyattr]
    #[pyclass(name = "wraps")]
    #[derive(Debug, PyPayload)]
    struct Pyo3Wraps {
        source: String,
        func_name: String,
    }

    impl Constructor for Pyo3Wraps {
        type Args = PyObjectRef;

        fn py_new(_cls: &Py<PyType>, func: Self::Args, vm: &VirtualMachine) -> PyResult<Self> {
            // Get function name
            let func_name = func
                .get_attr("__name__", vm)?
                .downcast::<rustpython_vm::builtins::PyStr>()
                .map_err(|_| vm.new_type_error("function must have __name__".to_owned()))?
                .as_str()
                .to_owned();

            // Get source using inspect.getsource(func)
            let inspect = vm.import("inspect", 0)?;
            let getsource = inspect.get_attr("getsource", vm)?;
            let source_obj = getsource.call((func.clone(),), vm)?;
            let source_full = source_obj
                .downcast::<rustpython_vm::builtins::PyStr>()
                .map_err(|_| vm.new_type_error("getsource did not return str".to_owned()))?
                .as_str()
                .to_owned();

            // Strip decorator lines from source (lines starting with @)
            // Find the first line that starts with 'def ' or 'async def '
            let source = strip_decorators(&source_full);

            Ok(Self { source, func_name })
        }
    }

    /// Serialize a RustPython object to pickle bytes.
    fn rustpython_pickle_dumps(
        obj: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyRef<RustPyBytes>> {
        let pickle = vm.import("pickle", 0)?;
        let dumps = pickle.get_attr("dumps", vm)?;
        dumps
            .call((obj,), vm)?
            .downcast::<RustPyBytes>()
            .map_err(|_| vm.new_type_error("pickle.dumps did not return bytes".to_owned()))
    }

    /// Deserialize pickle bytes to a RustPython object.
    fn rustpython_pickle_loads(bytes: &[u8], vm: &VirtualMachine) -> PyResult {
        let pickle = vm.import("pickle", 0)?;
        let loads = pickle.get_attr("loads", vm)?;
        let bytes_obj = RustPyBytes::from(bytes.to_vec()).into_ref(&vm.ctx);
        loads.call((bytes_obj,), vm)
    }

    /// Strip decorator lines from function source code and dedent.
    /// Returns source starting from 'def' or 'async def', with common indentation removed.
    fn strip_decorators(source: &str) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let mut result_lines = Vec::new();
        let mut found_def = false;
        let mut base_indent = 0;

        for line in &lines {
            let trimmed = line.trim_start();
            if !found_def {
                if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                    found_def = true;
                    // Calculate base indentation from the def line
                    base_indent = line.len() - trimmed.len();
                    result_lines.push(*line);
                }
                // Skip decorator lines (starting with @) and blank lines before def
            } else {
                result_lines.push(*line);
            }
        }

        // Dedent all lines by base_indent
        result_lines
            .iter()
            .map(|line| {
                if line.len() >= base_indent
                    && line
                        .chars()
                        .take(base_indent)
                        .all(|c| c == ' ' || c == '\t')
                {
                    &line[base_indent..]
                } else if line.trim().is_empty() {
                    ""
                } else {
                    *line
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    impl Callable for Pyo3Wraps {
        type Args = FuncArgs;

        fn call(zelf: &Py<Self>, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
            // Pickle args and kwargs
            let args_tuple = vm.ctx.new_tuple(args.args);
            let kwargs_dict = PyDict::default().into_ref(&vm.ctx);
            for (key, value) in args.kwargs {
                kwargs_dict.set_item(&key, value, vm)?;
            }

            let pickled_args_bytes = rustpython_pickle_dumps(args_tuple.into(), vm)?;
            let pickled_kwargs_bytes = rustpython_pickle_dumps(kwargs_dict.into(), vm)?;

            // Call execute_impl()
            let result_bytes = execute_impl(
                &zelf.source,
                &zelf.func_name,
                pickled_args_bytes.as_bytes(),
                pickled_kwargs_bytes.as_bytes(),
                vm,
            )?;

            // Unpickle result
            rustpython_pickle_loads(&result_bytes, vm)
        }
    }

    impl Representable for Pyo3Wraps {
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            Ok(format!("<_cpython.wraps wrapper for '{}'>", zelf.func_name))
        }
    }

    #[pyclass(with(Constructor, Callable, Representable))]
    impl Pyo3Wraps {}

    /// Internal implementation for executing Python code in CPython.
    fn execute_impl(
        source: &str,
        func_name: &str,
        args_bytes: &[u8],
        kwargs_bytes: &[u8],
        vm: &VirtualMachine,
    ) -> PyResult<Vec<u8>> {
        // Build the CPython code to execute
        let pyo3_code = format!(
            r#"
import pickle as __pickle

# Unpickle arguments
__args__ = __pickle.loads(__pickled_args__)
__kwargs__ = __pickle.loads(__pickled_kwargs__)
# Execute the source code (defines the function)
{source}

# Call the function and pickle the result
__result__ = {func_name}(*__args__, **__kwargs__)
__pickled_result__ = __pickle.dumps(__result__, protocol=4)
"#,
            source = source,
            func_name = func_name,
        );

        // Execute in CPython via PyO3
        pyo3::Python::attach(|py| -> Result<Vec<u8>, PyErr> {
            // Create Python bytes for pickled data
            let py_args = Pyo3Bytes::new(py, args_bytes);
            let py_kwargs = Pyo3Bytes::new(py, kwargs_bytes);

            // Create globals dict with pickled args
            let globals = pyo3::types::PyDict::new(py);
            globals.set_item("__pickled_args__", &py_args)?;
            globals.set_item("__pickled_kwargs__", &py_kwargs)?;

            // Execute using compile + exec pattern
            let builtins = py.import("builtins")?;
            let compile = builtins.getattr("compile")?;
            let exec_fn = builtins.getattr("exec")?;

            // Compile the code
            let code = compile.call1((&pyo3_code, "<pyo3_bridge>", "exec"))?;

            // Execute with globals
            exec_fn.call1((code, &globals))?;

            // Get the pickled result
            let result = globals.get_item("__pickled_result__")?;
            let result = result.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result returned")
            })?;
            let result_bytes: &pyo3::Bound<'_, Pyo3Bytes> = result.cast()?;
            Ok(result_bytes.as_bytes().to_vec())
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))
    }

    /// Execute a Python function in CPython runtime.
    ///
    /// # Arguments
    /// * `source` - The complete source code of the function
    /// * `func_name` - The name of the function to call
    /// * `pickled_args` - Pickled positional arguments (bytes)
    /// * `pickled_kwargs` - Pickled keyword arguments (bytes)
    ///
    /// # Returns
    /// Pickled result from CPython (bytes)
    #[pyfunction]
    fn execute(
        source: PyStrRef,
        func_name: PyStrRef,
        pickled_args: PyBytesRef,
        pickled_kwargs: PyBytesRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyBytesRef> {
        let result_bytes = execute_impl(
            source.as_str(),
            func_name.as_str(),
            pickled_args.as_bytes(),
            pickled_kwargs.as_bytes(),
            vm,
        )?;
        Ok(RustPyBytes::from(result_bytes).into_ref(&vm.ctx))
    }

    /// Execute arbitrary Python code in CPython and return pickled result.
    ///
    /// # Arguments
    /// * `code` - Python code to execute (should assign result to `__result__`)
    ///
    /// # Returns
    /// Pickled result from CPython (bytes)
    #[pyfunction]
    fn eval_code(code: PyStrRef, vm: &VirtualMachine) -> PyResult<PyBytesRef> {
        let code_str = code.as_str();

        let wrapper_code = format!(
            r#"
import pickle
{code}
__pickled_result__ = pickle.dumps(__result__, protocol=4)
"#,
            code = code_str,
        );

        let result_bytes = pyo3::Python::attach(|py| -> Result<Vec<u8>, PyErr> {
            let globals = pyo3::types::PyDict::new(py);

            let builtins = py.import("builtins")?;
            let compile = builtins.getattr("compile")?;
            let exec_fn = builtins.getattr("exec")?;

            let code = compile.call1((&wrapper_code, "<pyo3_bridge>", "exec"))?;
            exec_fn.call1((code, &globals))?;

            let result = globals.get_item("__pickled_result__")?;
            let result = result.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No __result__ defined in code")
            })?;
            let result_bytes: &pyo3::Bound<'_, Pyo3Bytes> = result.cast()?;
            Ok(result_bytes.as_bytes().to_vec())
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        Ok(RustPyBytes::from(result_bytes).into_ref(&vm.ctx))
    }

    /// Pickle a CPython object to bytes.
    fn pickle_in_cpython(
        py: pyo3::Python<'_>,
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
    ) -> Result<Vec<u8>, PyErr> {
        let pickle = py.import("pickle")?;
        let pickled = pickle.call_method1("dumps", (obj, 4i32))?;
        let bytes: &pyo3::Bound<'_, Pyo3Bytes> = pickled.cast()?;
        Ok(bytes.as_bytes().to_vec())
    }

    /// Unpickle bytes in CPython.
    fn unpickle_in_cpython<'py>(
        py: pyo3::Python<'py>,
        bytes: &[u8],
    ) -> Result<pyo3::Bound<'py, pyo3::PyAny>, PyErr> {
        let pickle = py.import("pickle")?;
        pickle.call_method1("loads", (Pyo3Bytes::new(py, bytes),))
    }

    /// Metaclass for CPython types wrapped in RustPython.
    /// This is a subclass of `type`, so instances of Pyo3Type are types themselves.
    /// The actual CPython type is stored in __pyo3_obj__ attribute.
    #[pyattr]
    #[pyclass(name = "pyo3_type", base = PyType, module = "_pyo3")]
    #[derive(Debug)]
    #[repr(transparent)]
    struct Pyo3Type(PyType);

    /// Custom __new__ slot for Pyo3Type metaclass.
    /// Called when creating a new class with Pyo3Type as the metaclass (e.g., `class Foo(_SimpleCData)`).
    /// If any base has __pyo3_obj__, delegate class creation to CPython.
    fn pyo3_metatype_new(metatype: PyTypeRef, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
        // Parse args as (name, bases, namespace)
        if args.args.len() != 3 {
            return Err(vm.new_type_error(format!(
                "type.__new__() takes exactly 3 arguments ({} given)",
                args.args.len()
            )));
        }

        let bases = args.args[1]
            .downcast_ref::<rustpython_vm::builtins::PyTuple>()
            .ok_or_else(|| vm.new_type_error("bases must be a tuple".to_owned()))?;
        let namespace = &args.args[2];

        // Check if any base has __pyo3_obj__ - if so, create class in CPython
        for base in bases.iter() {
            if let Some(base_type) = base.downcast_ref::<PyType>()
                && let Some(pyo3_obj) = base_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
            {
                // Base has CPython type - create class in CPython
                let class = create_class_in_cpython(&args, &pyo3_ref.py_obj, vm)?;

                // Handle __classcell__ - set it to the new class for super() to work
                // If __classcell__ exists, it's a cell object that needs to be set to the new class
                if let Ok(classcell) = namespace.get_item(vm.ctx.intern_str("__classcell__"), vm) {
                    // Check if it's a cell type and set its value via cell_contents property
                    if classcell.class().is(vm.ctx.types.cell_type) {
                        classcell
                            .set_attr(vm.ctx.intern_str("cell_contents"), class.clone(), vm)
                            .ok();
                    }
                }

                return Ok(class);
            }
        }

        // No CPython bases - use default type.__new__ behavior
        let type_type = vm.ctx.types.type_type;
        type_type
            .slots
            .new
            .load()
            .expect("type should have __new__")(metatype, args, vm)
    }

    #[pyclass(flags(BASETYPE))]
    impl Pyo3Type {
        /// Get attribute from type - delegates to CPython for __pyo3_obj__ types.
        #[pymethod]
        fn __getattribute__(zelf: PyTypeRef, name: PyStrRef, vm: &VirtualMachine) -> PyResult {
            // First check if the attribute is __pyo3_obj__ itself or a RustPython attribute
            let name_str = name.as_str();

            // Don't intercept special attributes that should stay in RustPython
            if name_str == PYO3_OBJ_ATTR
                || name_str == "__class__"
                || name_str == "__name__"
                || name_str == "__qualname__"
                || name_str == "__module__"
                || name_str == "__bases__"
                || name_str == "__mro__"
                || name_str == "__dict__"
            {
                // Use default type.__getattribute__
                let type_type = vm.ctx.types.type_type;
                return vm.call_method(type_type.as_object(), "__getattribute__", (zelf, name));
            }

            // Check if this type has a CPython object reference
            if let Some(pyo3_obj) = zelf.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
            {
                // Delegate attribute lookup to CPython
                return pyo3_getattr_impl(&pyo3_ref.py_obj, name_str, vm);
            }

            // No __pyo3_obj__ - use default type.__getattribute__
            let type_type = vm.ctx.types.type_type;
            vm.call_method(type_type.as_object(), "__getattribute__", (zelf, name))
        }

        /// Support type * int for array type creation (e.g., CHAR * 10)
        /// Delegates to CPython's type.__mul__ which creates array types.
        #[pymethod]
        fn __mul__(zelf: PyTypeRef, n: PyObjectRef, vm: &VirtualMachine) -> PyResult {
            // Check if this type has a CPython object reference
            if let Some(pyo3_obj) = zelf.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
            {
                // Delegate __mul__ to CPython type
                return pyo3_binary_op_with_pyo3ref(&pyo3_ref.py_obj, &n, "__mul__", vm);
            }

            // No __pyo3_obj__ - return NotImplemented
            Ok(vm.ctx.not_implemented())
        }

        /// Support int * type (reverse multiplication)
        #[pymethod]
        fn __rmul__(zelf: PyTypeRef, n: PyObjectRef, vm: &VirtualMachine) -> PyResult {
            // Same as __mul__ for commutative operation
            Pyo3Type::__mul__(zelf, n, vm)
        }
    }

    /// Key for storing CPython object in type's __dict__
    const PYO3_OBJ_ATTR: &str = "__pyo3_obj__";

    /// Create a Pyo3Ref from a pyo3 object, attempting to pickle it.
    fn create_pyo3_object(py: pyo3::Python<'_>, obj: &pyo3::Bound<'_, pyo3::PyAny>) -> Pyo3Ref {
        let pickled = pickle_in_cpython(py, obj).ok();
        Pyo3Ref {
            py_obj: obj.clone().unbind(),
            pickled,
        }
    }

    /// Custom __getattribute__ slot for Pyo3Type instances.
    /// Delegates to CPython if the type has __pyo3_obj__.
    fn pyo3_type_getattro(obj: &PyObject, name: &Py<PyStr>, vm: &VirtualMachine) -> PyResult {
        let name_str = name.as_str();

        // Special attributes that should stay in RustPython (not delegated to CPython)
        // __pyo3_obj__ and __pyo3_subtype__ are internal attributes for pyo3 bridge
        if name_str == PYO3_OBJ_ATTR || name_str == PYO3_SUBTYPE_ATTR {
            // Use default type getattro to get these from RustPython
            let type_type = vm.ctx.types.type_type;
            if let Some(getattro) = type_type.slots.getattro.load() {
                return getattro(obj, name, vm);
            }
            return obj.generic_getattr(name, vm);
        }

        // Special handling for __flags__ on wrapped CPython types
        // CPython types have specific tp_flags that need to be returned as integers
        if name_str == "__flags__"
            && let Some(py_type) = obj.downcast_ref::<PyType>()
            && let Some(pyo3_obj) = py_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            let flags = pyo3::Python::attach(|py| -> Result<i64, pyo3::PyErr> {
                let type_obj = pyo3_ref.py_obj.bind(py);
                let flags = type_obj.getattr("__flags__")?;
                flags.extract::<i64>()
            });
            if let Ok(flags) = flags {
                return Ok(vm.ctx.new_int(flags).into());
            }
        }

        // If this is a type with __pyo3_obj__, delegate attribute lookup to CPython
        if let Some(py_type) = obj.downcast_ref::<PyType>()
            && let Some(pyo3_obj) = py_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            return pyo3_getattr_impl(&pyo3_ref.py_obj, name_str, vm);
        }

        // Fall back to type's getattro
        let type_type = vm.ctx.types.type_type;
        if let Some(getattro) = type_type.slots.getattro.load() {
            return getattro(obj, name, vm);
        }

        // Default to generic object getattro
        obj.generic_getattr(name, vm)
    }

    /// Custom __call__ slot for Pyo3Type metaclass.
    /// Called when a class (instance of Pyo3Type) is called to create an instance.
    fn pyo3_type_call(zelf: &PyObject, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
        // Get the type
        let zelf_type = zelf
            .downcast_ref::<PyType>()
            .ok_or_else(|| vm.new_type_error("expected type"))?;

        // Check if this type has a CPython object reference
        if let Some(pyo3_obj) = zelf_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            // Delegate to CPython
            return pyo3_call_impl(&pyo3_ref.py_obj, args, vm);
        }

        // Check if there's a cached CPython subtype
        if let Some(subtype) = zelf_type.get_attr(vm.ctx.intern_str(PYO3_SUBTYPE_ATTR))
            && let Some(pyo3_ref) = subtype.downcast_ref::<Pyo3Ref>()
        {
            return pyo3_call_impl(&pyo3_ref.py_obj, args, vm);
        }

        // No __pyo3_obj__ - use default type.__call__ behavior
        let type_type = vm.ctx.types.type_type;
        if let Some(call) = type_type.slots.call.load() {
            return call(zelf, args, vm);
        }

        Err(vm.new_type_error(format!("'{}' object is not callable", zelf.class().name())))
    }

    /// Custom setattro slot for Pyo3Type.
    /// Delegates to CPython if the type has __pyo3_obj__.
    fn pyo3_type_setattro(
        obj: &PyObject,
        name: &Py<PyStr>,
        value: PySetterValue,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let name_str = name.as_str();

        // Internal attributes (__pyo3_obj__, __pyo3_subtype__) are handled by RustPython
        if name_str == PYO3_OBJ_ATTR || name_str == PYO3_SUBTYPE_ATTR {
            let type_type = vm.ctx.types.type_type;
            if let Some(setattro) = type_type.slots.setattro.load() {
                return setattro(obj, name, value, vm);
            }
            return Err(vm.new_attribute_error("cannot set attribute".to_owned()));
        }

        // If __pyo3_obj__ exists, delegate to CPython
        if let Some(py_type) = obj.downcast_ref::<PyType>()
            && let Some(pyo3_obj) = py_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            return pyo3::Python::attach(|py| -> Result<(), pyo3::PyErr> {
                let cpython_obj = pyo3_ref.py_obj.bind(py);
                match value {
                    PySetterValue::Assign(val) => {
                        let pyo3_val = to_pyo3_object(&val, vm).map_err(|e| {
                            pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to convert value: {:?}",
                                e
                            ))
                        })?;
                        cpython_obj.setattr(name_str, pyo3_val.to_pyo3(py)?)?;
                    }
                    PySetterValue::Delete => {
                        cpython_obj.delattr(name_str)?;
                    }
                }
                Ok(())
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm));
        }

        // Default: use type's setattro
        let type_type = vm.ctx.types.type_type;
        if let Some(setattro) = type_type.slots.setattro.load() {
            return setattro(obj, name, value, vm);
        }
        Err(vm.new_attribute_error(format!("cannot set '{}' attribute", name_str)))
    }

    /// Custom __new__ slot for Pyo3Type instances.
    /// Delegates to CPython if the type or its bases have __pyo3_obj__.
    fn pyo3_type_new(subtype: PyTypeRef, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
        // Check if this type has __pyo3_obj__
        if let Some(pyo3_obj) = subtype.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            return pyo3_call_impl(&pyo3_ref.py_obj, args, vm);
        }

        // Check if any base has __pyo3_obj__ (e.g., class CFunctionType(_CFuncPtr))
        // In this case, we need to create a corresponding CPython class first
        if let Some(bases) = subtype.get_attr(vm.ctx.intern_str("__bases__"))
            && let Some(bases_tuple) = bases.downcast_ref::<rustpython_vm::builtins::PyTuple>()
        {
            for base in bases_tuple.iter() {
                if let Some(base_type) = base.downcast_ref::<PyType>()
                    && let Some(pyo3_obj) = base_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                    && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
                {
                    // Create a CPython subclass dynamically and cache it
                    let cpython_subtype =
                        get_or_create_cpython_subtype(&subtype, &pyo3_ref.py_obj, vm)?;
                    return pyo3_call_impl(&cpython_subtype, args, vm);
                }
            }
        }

        // Default to object.__new__
        let object_new = vm
            .ctx
            .types
            .object_type
            .slots
            .new
            .load()
            .expect("object should have __new__");
        object_new(subtype, args, vm)
    }

    /// Cache key for CPython subtypes
    const PYO3_SUBTYPE_ATTR: &str = "__pyo3_subtype__";

    /// Get or create a CPython subtype for a RustPython class that inherits from a CPython type.
    fn get_or_create_cpython_subtype(
        subtype: &PyTypeRef,
        base_py_obj: &pyo3::Py<pyo3::PyAny>,
        vm: &VirtualMachine,
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        // Check if we already created a CPython subtype for this
        if let Some(cached) = subtype.get_attr(vm.ctx.intern_str(PYO3_SUBTYPE_ATTR))
            && let Some(pyo3_ref) = cached.downcast_ref::<Pyo3Ref>()
        {
            return Ok(pyo3::Python::attach(|py| pyo3_ref.py_obj.clone_ref(py)));
        }

        // Get the class name
        let name = subtype.name().to_string();

        // Collect relevant class attributes to pass to CPython
        let attr_names = [
            "_argtypes_",
            "_restype_",
            "_flags_",
            "_type_",
            "_length_",
            "_fields_",
        ];

        // Collect attribute values as owned CPython objects
        let mut collected_attrs: Vec<(String, pyo3::Py<pyo3::PyAny>)> = Vec::new();
        for attr_name in attr_names {
            if let Some(val) = subtype.get_attr(vm.ctx.intern_str(attr_name))
                && let Ok(pyo3_val) = to_pyo3_object(&val, vm)
            {
                let owned = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
                    Ok(pyo3_val.to_pyo3(py)?.unbind())
                });
                if let Ok(py_obj) = owned {
                    collected_attrs.push((attr_name.to_string(), py_obj));
                }
            }
        }

        let cpython_subtype = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
            let base = base_py_obj.bind(py);

            // Create a new dict for the CPython class
            let class_dict = pyo3::types::PyDict::new(py);

            // Copy over ctypes-specific attributes
            for (attr_name, py_obj) in &collected_attrs {
                class_dict.set_item(attr_name.as_str(), py_obj.bind(py))?;
            }

            // Create the CPython subclass using base's metaclass
            // metaclass(name, (base,), dict)
            let metaclass = base.get_type();
            let bases = pyo3::types::PyTuple::new(py, [base])?;
            let new_type = metaclass.call1((&name, bases, class_dict))?;

            Ok(new_type.unbind())
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        // Cache the CPython subtype
        let cpython_subtype_for_cache = pyo3::Python::attach(|py| cpython_subtype.clone_ref(py));
        let pyo3_ref = Pyo3Ref {
            py_obj: cpython_subtype_for_cache,
            pickled: None,
        };
        subtype.set_attr(
            vm.ctx.intern_str(PYO3_SUBTYPE_ATTR),
            pyo3_ref.into_ref(&vm.ctx).into(),
        );

        Ok(cpython_subtype)
    }

    /// Create a class in CPython when inheriting from a CPython type.
    /// Called from Pyo3Type.__call__ when creating a class like `class Foo(CPythonBase): ...`
    fn create_class_in_cpython(
        args: &FuncArgs,
        base_py_obj: &pyo3::Py<pyo3::PyAny>,
        vm: &VirtualMachine,
    ) -> PyResult {
        // Extract (name, bases, namespace) from args
        if args.args.len() < 3 {
            return Err(vm.new_type_error("type() takes 3 arguments".to_owned()));
        }

        let name = args.args[0]
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("type name must be a string".to_owned()))?
            .as_str()
            .to_owned();

        let bases = args.args[1]
            .downcast_ref::<rustpython_vm::builtins::PyTuple>()
            .ok_or_else(|| vm.new_type_error("bases must be a tuple".to_owned()))?;

        // Convert namespace to CPython dict
        let namespace = &args.args[2];

        // Collect namespace items that can be passed to CPython
        let namespace_items: Vec<(String, PyObjectRef)> =
            if let Ok(mapping) = namespace.try_mapping(vm) {
                let keys = mapping.keys(vm)?;
                let mut items = Vec::new();
                let keys_iter = keys.get_iter(vm)?;
                loop {
                    use rustpython_vm::protocol::PyIterReturn;
                    match keys_iter.next(vm)? {
                        PyIterReturn::Return(key) => {
                            if let Some(key_str) = key.downcast_ref::<PyStr>() {
                                let key_name = key_str.as_str().to_owned();
                                let value = mapping.as_ref().get_item(key_str.as_str(), vm)?;
                                items.push((key_name, value));
                            }
                        }
                        PyIterReturn::StopIteration(_) => break,
                    }
                }
                items
            } else {
                Vec::new()
            };

        // Create the class in CPython
        let result = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let base = base_py_obj.bind(py);

            // Build CPython bases tuple - get __pyo3_obj__ from each base
            let mut cpython_bases = Vec::new();
            for base_obj in bases.iter() {
                if let Some(base_type) = base_obj.downcast_ref::<PyType>() {
                    if let Some(pyo3_obj) = base_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                        && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
                    {
                        cpython_bases.push(pyo3_ref.py_obj.bind(py).clone());
                    } else {
                        // RustPython base without __pyo3_obj__ - skip or error
                        // For now, we just use the CPython base we found
                    }
                }
            }

            // If no CPython bases found, use the provided base
            if cpython_bases.is_empty() {
                cpython_bases.push(base.clone());
            }

            let bases_tuple = pyo3::types::PyTuple::new(py, &cpython_bases)?;

            // Create namespace dict in CPython
            // Callables are wrapped with _MethodDescriptor for proper method binding
            let class_dict = pyo3::types::PyDict::new(py);
            for (key, value) in &namespace_items {
                // Convert value to CPython (callables become descriptor-wrapped closures)
                if let Ok(pyo3_val) = to_pyo3_object(value, vm)
                    && let Ok(py_val) = pyo3_val.to_pyo3(py)
                {
                    class_dict.set_item(key.as_str(), py_val)?;
                }
            }

            // Get the metaclass from the base type
            let metaclass = base.get_type();

            // Create the new type: metaclass(name, bases, dict)
            let new_type = metaclass.call1((&name, bases_tuple, class_dict))?;

            Ok(create_pyo3_object(py, &new_type))
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        // Wrap the CPython type as a Pyo3Type
        pyo3_to_rustpython(result, vm)
    }

    /// nb_multiply slot for Pyo3Type instances.
    /// Delegates type * int multiplication to CPython (e.g., CHAR * 10 creates array type).
    fn pyo3_type_multiply(a: &PyObject, b: &PyObject, vm: &VirtualMachine) -> PyResult {
        // a is a type with __pyo3_obj__ attribute containing the CPython type
        if let Some(type_obj) = a.downcast_ref::<PyType>()
            && let Some(pyo3_obj) = type_obj.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
            && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
        {
            return pyo3_binary_op_with_pyo3ref(&pyo3_ref.py_obj, b, "__mul__", vm);
        }
        Ok(vm.ctx.not_implemented())
    }

    /// Map CPython builtin types to RustPython native types.
    /// Uses pointer-based lookup for reliable type identity.
    /// Returns Some(native_type) if the CPython type should be mapped to a native type,
    /// None if it should be wrapped as a Pyo3Type.
    fn map_cpython_builtin_to_native(
        py: pyo3::Python<'_>,
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        vm: &VirtualMachine,
    ) -> Option<PyTypeRef> {
        // Initialize the builtin type map if not already done
        init_builtin_type_map(py);

        // Look up by CPython type pointer
        let ptr = obj.as_ptr() as isize;
        let map = BUILTIN_TYPE_MAP.lock().unwrap();
        if let Some(&id) = map.get(&ptr) {
            return Some(resolve_builtin_type_id(id, vm));
        }

        None
    }

    /// Create a wrapper for a CPython type.
    /// Returns a RustPython type (instance of Pyo3Type) that wraps the CPython type.
    fn create_pyo3_type(
        py: pyo3::Python<'_>,
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        vm: &VirtualMachine,
    ) -> PyResult<PyTypeRef> {
        use crossbeam_utils::atomic::AtomicCell;
        use pyo3::types::PyTupleMethods as _;
        use rustpython_vm::class::PyClassImpl;
        use rustpython_vm::types::{PyTypeFlags, PyTypeSlots};

        // Check cache first - ensures type identity
        let ptr = obj.as_ptr() as isize;
        if let Some(cached) = TYPE_CACHE.lock().unwrap().get(&ptr) {
            return Ok(cached.clone());
        }

        // Map CPython builtin types to RustPython native types
        // This ensures type identity for all builtin types
        if let Some(native_type) = map_cpython_builtin_to_native(py, obj, vm) {
            return Ok(native_type);
        }

        // Get the name of the CPython type
        let name: String = obj
            .getattr("__name__")
            .map_err(|e| vm.new_attribute_error(format!("CPython type has no __name__: {}", e)))?
            .extract()
            .map_err(|e| vm.new_type_error(format!("__name__ is not a string: {}", e)))?;

        // Create a Pyo3Ref to store the CPython type
        let pyo3_ref = create_pyo3_object(py, obj);
        let pyo3_ref_obj: PyObjectRef = pyo3_ref.into_ref(&vm.ctx).into();

        // Get Pyo3Type as the metaclass
        let pyo3_type_metaclass = Pyo3Type::make_class(&vm.ctx);

        // Set custom slots on Pyo3Type metaclass (only needs to be done once)
        // __new__: intercepts class creation to delegate to CPython if needed
        // __getattribute__: intercepts attribute access on types to delegate to CPython
        // __mul__: supports type * int for array type creation (e.g., CHAR * 10)
        pyo3_type_metaclass.slots.new.store(Some(pyo3_metatype_new));
        pyo3_type_metaclass
            .slots
            .getattro
            .store(Some(pyo3_type_getattro));
        pyo3_type_metaclass
            .slots
            .as_number
            .multiply
            .store(Some(pyo3_type_multiply));
        pyo3_type_metaclass.slots.call.store(Some(pyo3_type_call));
        pyo3_type_metaclass
            .slots
            .setattro
            .store(Some(pyo3_type_setattro));

        // Create slots with HEAPTYPE, BASETYPE flags and custom new/getattro/setattro slots
        let mut slots = PyTypeSlots::default();
        slots.flags = PyTypeFlags::HEAPTYPE | PyTypeFlags::BASETYPE;
        slots.new = AtomicCell::new(Some(pyo3_type_new));
        slots.getattro = AtomicCell::new(Some(pyo3_type_getattro));
        slots.setattro = AtomicCell::new(Some(pyo3_type_setattro));

        // Get CPython base classes and convert them to RustPython types
        let bases: Vec<PyTypeRef> = if let Ok(bases_obj) = obj.getattr("__bases__")
            && let Ok(bases_tuple) = bases_obj.cast::<pyo3::types::PyTuple>()
        {
            let mut result = Vec::new();
            for base in bases_tuple.iter() {
                // Check if base is a CPython builtin type that maps to native
                if let Some(native_type) = map_cpython_builtin_to_native(py, &base, vm) {
                    result.push(native_type);
                } else if base.is_instance_of::<pyo3::types::PyType>() {
                    // Recursively convert base type (cache handles cycles)
                    match create_pyo3_type(py, &base, vm) {
                        Ok(base_type) => result.push(base_type),
                        Err(_) => result.push(vm.ctx.types.object_type.to_owned()),
                    }
                }
            }
            if result.is_empty() {
                vec![vm.ctx.types.object_type.to_owned()]
            } else {
                result
            }
        } else {
            vec![vm.ctx.types.object_type.to_owned()]
        };

        // Create a new type with Pyo3Type as metaclass
        let new_type = PyType::new_heap(
            &name,
            bases,
            Default::default(), // Empty attributes - will be populated below
            slots,
            pyo3_type_metaclass,
            &vm.ctx,
        )
        .map_err(|e| vm.new_type_error(e))?;

        // Store the CPython object reference
        new_type.set_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR), pyo3_ref_obj);

        // Copy CPython type's __dict__ entries to RustPython type's attributes
        // This allows super() to find methods via get_direct_attr
        if let Ok(type_dict) = obj.getattr("__dict__")
            && let Ok(dict_items) = type_dict.call_method0("items")
            && let Ok(iter) = dict_items.try_iter()
        {
            for item in iter {
                if let Ok(item) = item
                    && let Ok(tuple) = item.cast::<pyo3::types::PyTuple>()
                    && tuple.len() == 2
                    && let (Ok(key), Ok(value)) = (tuple.get_item(0), tuple.get_item(1))
                    && let Ok(key_str) = key.extract::<String>()
                {
                    // Skip internal attributes
                    if key_str.starts_with("__pyo3") {
                        continue;
                    }
                    // Wrap the CPython value as Pyo3Ref
                    let pyo3_value = create_pyo3_object(py, &value);
                    let pyo3_value_obj: PyObjectRef = pyo3_value.into_ref(&vm.ctx).into();
                    new_type.set_attr(vm.ctx.intern_str(key_str), pyo3_value_obj);
                }
            }
        }

        // Cache the type for identity preservation
        TYPE_CACHE.lock().unwrap().insert(ptr, new_type.clone());

        Ok(new_type)
    }

    /// Check if a CPython object is a type
    fn is_cpython_type(py_obj: &pyo3::Py<pyo3::PyAny>) -> bool {
        pyo3::Python::attach(|py| {
            let obj = py_obj.bind(py);
            obj.is_instance_of::<pyo3::types::PyType>()
        })
    }

    /// Convert a Pyo3Ref to RustPython object.
    /// If the object is a CPython type, wraps it in Pyo3Type (must check first to preserve __pyo3_obj__).
    /// If pickled bytes exist, tries to unpickle to native RustPython object.
    /// Falls back to returning the Pyo3Ref wrapper.
    fn pyo3_to_rustpython(pyo3_obj: Pyo3Ref, vm: &VirtualMachine) -> PyResult {
        // IMPORTANT: Check CPython type FIRST, before unpickling.
        // CPython types (like ctypes c_char, Structure, etc.) are picklable, but if we
        // unpickle them, we get a RustPython type without __pyo3_obj__, which breaks
        // special methods like __mul__ (e.g., c_char * 6 for array types).
        if is_cpython_type(&pyo3_obj.py_obj) {
            let type_ref = pyo3::Python::attach(|py| {
                let obj = pyo3_obj.py_obj.bind(py);
                create_pyo3_type(py, obj, vm)
            })?;
            return Ok(type_ref.into());
        }

        // IMPORTANT: ctypes instances (Structure, Union, Array, etc.) should NOT be unpickled.
        // Unpickling ctypes instances corrupts their buffer pointers because _ctypes._unpickle
        // cannot properly restore the memory buffer across the pyo3 boundary.
        // Keep them as Pyo3Ref so field access goes through CPython.
        if is_ctypes_instance(&pyo3_obj.py_obj) {
            return Ok(pyo3_obj.into_ref(&vm.ctx).into());
        }

        // Try unpickling to native RustPython object (for non-type values)
        if let Some(ref bytes) = pyo3_obj.pickled
            && let Ok(unpickled) = rustpython_pickle_loads(bytes, vm)
        {
            return Ok(unpickled);
        }
        // Unpickle failed (e.g., numpy arrays need numpy module)
        // Fall through

        // Default: return as Pyo3Ref
        Ok(pyo3_obj.into_ref(&vm.ctx).into())
    }

    /// Check if a CPython object is a ctypes instance (Structure, Union, Array, etc.)
    /// These should not be unpickled as it corrupts their memory buffers.
    fn is_ctypes_instance(py_obj: &pyo3::Py<pyo3::PyAny>) -> bool {
        pyo3::Python::attach(|py| {
            let obj = py_obj.bind(py);

            // Try to import _ctypes and check if obj is instance of _CData
            // _CData is the base class for all ctypes data types (Structure, Union, Array, etc.)
            if let Ok(ctypes_module) = py.import("_ctypes")
                && let Ok(cdata_class) = ctypes_module.getattr("_CData")
                && let Ok(is_instance) = obj.is_instance(&cdata_class)
            {
                return is_instance;
            }

            // Fallback: check module name
            if let Ok(type_obj) = obj.get_type().getattr("__module__")
                && let Ok(module_name) = type_obj.extract::<String>()
                && (module_name == "_ctypes" || module_name.starts_with("ctypes"))
                && !obj.is_instance_of::<pyo3::types::PyType>()
            {
                return true;
            }

            false
        })
    }

    /// Get attribute from a CPython object
    fn pyo3_getattr_impl(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        name: &str,
        vm: &VirtualMachine,
    ) -> PyResult {
        let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let obj = py_obj.bind(py);
            let attr = obj.getattr(name)?;
            Ok(create_pyo3_object(py, &attr))
        })
        .map_err(|e| vm.new_attribute_error(format!("CPython getattr error: {}", e)))?;

        pyo3_to_rustpython(pyo3_obj, vm)
    }

    /// Call a CPython object
    fn pyo3_call_impl(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        args: FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult {
        // Convert each arg using to_pyo3_object (handles Pyo3Ref specially)
        let converted_args: Vec<ToPyo3Ref<'_>> = args
            .args
            .iter()
            .map(|arg| to_pyo3_object(arg, vm))
            .collect::<PyResult<Vec<_>>>()?;

        // Convert kwargs
        let converted_kwargs: Vec<(String, ToPyo3Ref<'_>)> = args
            .kwargs
            .iter()
            .map(|(k, v)| Ok((k.clone(), to_pyo3_object(v, vm)?)))
            .collect::<PyResult<Vec<_>>>()?;

        let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let obj = py_obj.bind(py);

            // Build args tuple in CPython
            let args_list: Vec<pyo3::Bound<'_, pyo3::PyAny>> = converted_args
                .iter()
                .map(|arg| arg.to_pyo3(py))
                .collect::<Result<Vec<_>, _>>()?;
            let args_tuple = pyo3::types::PyTuple::new(py, &args_list)?;

            // Build kwargs dict in CPython
            let kwargs_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &converted_kwargs {
                kwargs_dict.set_item(k, v.to_pyo3(py)?)?;
            }

            // Call the object
            let call_result = obj.call(&args_tuple, Some(&kwargs_dict))?;

            Ok(create_pyo3_object(py, &call_result))
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        pyo3_to_rustpython(pyo3_obj, vm)
    }

    /// Keeps RustPython buffer alive while CPython uses it via shared memory.
    struct BufferGuard {
        /// The RustPython object that owns the buffer
        _owner: PyObjectRef,
        /// The buffer reference (prevents buffer release)
        _buffer: PyBuffer,
    }

    /// Represents an object to be passed into CPython.
    /// Either already a CPython object (Native/OwnedNative), pickled RustPython object (Pickled),
    /// or a shared buffer view (SharedBuffer).
    enum ToPyo3Ref<'a> {
        Native(&'a pyo3::Py<pyo3::PyAny>),
        OwnedNative(pyo3::Py<pyo3::PyAny>),
        Pickled(PyRef<RustPyBytes>),
        /// Shared buffer with CPython memoryview pointing to RustPython memory
        SharedBuffer {
            memoryview: pyo3::Py<pyo3::PyAny>,
            _guard: Arc<BufferGuard>,
        },
    }

    impl ToPyo3Ref<'_> {
        fn to_pyo3<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> Result<pyo3::Bound<'py, pyo3::PyAny>, PyErr> {
            match self {
                ToPyo3Ref::Native(obj) => Ok(obj.bind(py).clone()),
                ToPyo3Ref::OwnedNative(obj) => Ok(obj.bind(py).clone()),
                ToPyo3Ref::Pickled(bytes) => unpickle_in_cpython(py, bytes.as_bytes()),
                ToPyo3Ref::SharedBuffer { memoryview, .. } => Ok(memoryview.bind(py).clone()),
            }
        }
    }

    /// Create a CPython memoryview that shares memory with a RustPython buffer.
    /// This enables zero-copy buffer passing for ctypes.from_buffer() and similar APIs.
    fn create_shared_buffer_view(
        obj: &PyObject,
        buffer: PyBuffer,
        vm: &VirtualMachine,
    ) -> PyResult<ToPyo3Ref<'static>> {
        // Only support contiguous buffers
        if !buffer.desc.is_contiguous() {
            // Fall back to pickle for non-contiguous buffers
            let pickled = rustpython_pickle_dumps(obj.to_owned(), vm)?;
            return Ok(ToPyo3Ref::Pickled(pickled));
        }

        // Get buffer info
        let readonly = buffer.desc.readonly;
        let len = buffer.desc.len;

        // Create guard to keep RustPython object and buffer alive
        let guard = Arc::new(BufferGuard {
            _owner: obj.to_owned(),
            _buffer: buffer,
        });

        // Get the raw pointer from the buffer (must be done after creating guard)
        let ptr = {
            let bytes = guard
                ._buffer
                .as_contiguous()
                .ok_or_else(|| vm.new_type_error("Buffer is not contiguous".to_owned()))?;
            bytes.as_ptr()
        };

        // Clone Arc for the capsule destructor
        let guard_for_capsule = Arc::clone(&guard);

        // Create CPython memoryview with a wrapper to hold lifetime guard
        let memoryview = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
            use pyo3::ffi;
            use std::ffi::c_void;

            // Determine flags based on readonly
            let flags = if readonly {
                ffi::PyBUF_READ
            } else {
                ffi::PyBUF_WRITE
            };

            // Create memoryview from memory pointer
            let memoryview_ptr =
                unsafe { ffi::PyMemoryView_FromMemory(ptr as *mut i8, len as isize, flags) };

            if memoryview_ptr.is_null() {
                return Err(PyErr::fetch(py));
            }

            // Create capsule to prevent deallocation of RustPython buffer
            // The destructor drops the Arc<BufferGuard> when capsule is garbage collected
            let guard_box = Box::new(guard_for_capsule);

            extern "C" fn destructor(capsule: *mut ffi::PyObject) {
                unsafe {
                    let ptr = ffi::PyCapsule_GetPointer(capsule, std::ptr::null());
                    if !ptr.is_null() {
                        // Drop the Arc<BufferGuard>
                        drop(Box::from_raw(ptr as *mut Arc<BufferGuard>));
                    }
                }
            }

            let capsule_ptr = unsafe {
                ffi::PyCapsule_New(
                    Box::into_raw(guard_box) as *mut c_void,
                    std::ptr::null(),
                    Some(destructor),
                )
            };

            if capsule_ptr.is_null() {
                unsafe { ffi::Py_DECREF(memoryview_ptr) };
                return Err(PyErr::fetch(py));
            }

            // Store the capsule (which holds the guard) in BUFFER_GUARDS to keep it alive.
            // This ensures the RustPython buffer remains valid while CPython uses the memoryview.
            // Note: This is a simplified implementation that leaks memory - guards are never released.
            // Full buffer export tracking would require implementing buffer protocol on a custom type.
            let capsule_py: pyo3::Py<pyo3::PyAny> =
                unsafe { pyo3::Py::from_owned_ptr(py, capsule_ptr) };
            let memoryview_id = memoryview_ptr as isize;
            BUFFER_GUARDS
                .lock()
                .unwrap()
                .insert(memoryview_id, capsule_py);

            Ok(unsafe { pyo3::Py::from_owned_ptr(py, memoryview_ptr) })
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        Ok(ToPyo3Ref::SharedBuffer {
            memoryview,
            _guard: guard,
        })
    }

    /// Convert a RustPython object to ToPyo3Ref for passing into CPython
    fn to_pyo3_object<'a>(obj: &'a PyObject, vm: &VirtualMachine) -> PyResult<ToPyo3Ref<'a>> {
        // Check if it's a Pyo3Ref
        if let Some(pyo3_obj) = obj.downcast_ref::<Pyo3Ref>() {
            return Ok(ToPyo3Ref::Native(&pyo3_obj.py_obj));
        }

        // Check if it's a Pyo3Type (PyType with __pyo3_obj__ or __pyo3_subtype__)
        if let Some(py_type) = obj.downcast_ref::<PyType>() {
            // First check for __pyo3_obj__ (directly wrapped CPython types)
            if let Some(pyo3_obj) = py_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
            {
                let cloned = pyo3::Python::attach(|py| pyo3_ref.py_obj.clone_ref(py));
                return Ok(ToPyo3Ref::OwnedNative(cloned));
            }
            // Then check for __pyo3_subtype__ (RustPython subclasses of CPython types)
            if let Some(pyo3_obj) = py_type.get_attr(vm.ctx.intern_str(PYO3_SUBTYPE_ATTR))
                && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
            {
                let cloned = pyo3::Python::attach(|py| pyo3_ref.py_obj.clone_ref(py));
                return Ok(ToPyo3Ref::OwnedNative(cloned));
            }
            // Check if any base has __pyo3_obj__ - if so, create a subtype
            if let Some(bases) = py_type.get_attr(vm.ctx.intern_str("__bases__"))
                && let Some(bases_tuple) = bases.downcast_ref::<rustpython_vm::builtins::PyTuple>()
            {
                for base in bases_tuple.iter() {
                    if let Some(base_type) = base.downcast_ref::<PyType>()
                        && let Some(pyo3_obj) = base_type.get_attr(vm.ctx.intern_str(PYO3_OBJ_ATTR))
                        && let Some(pyo3_ref) = pyo3_obj.downcast_ref::<Pyo3Ref>()
                    {
                        // Create CPython subtype for this RustPython subclass
                        let cpython_subtype = get_or_create_cpython_subtype(
                            &py_type.to_owned(),
                            &pyo3_ref.py_obj,
                            vm,
                        )?;
                        return Ok(ToPyo3Ref::OwnedNative(cpython_subtype));
                    }
                }
            }
        }

        // Check if it's a CDLL-like object with _handle attribute
        // This is needed for ctypes to work - CDLL instances need to be passed to CPython
        if let Ok(handle_attr) = obj.get_attr(vm.ctx.intern_str("_handle"), vm) {
            // Convert _handle to CPython integer
            let handle_pyo3 = to_pyo3_object(&handle_attr, vm)?;
            // Create a CPython object with just the _handle attribute
            let cpython_dll = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
                let handle_py = handle_pyo3.to_pyo3(py)?;
                // Create a simple namespace object with _handle
                let types_module = py.import("types")?;
                let simple_namespace = types_module.getattr("SimpleNamespace")?;
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("_handle", handle_py)?;
                let dll_proxy = simple_namespace.call((), Some(&kwargs))?;
                Ok(dll_proxy.unbind())
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm))?;
            return Ok(ToPyo3Ref::OwnedNative(cpython_dll));
        }

        // Check if it's a tuple containing Pyo3 objects
        if let Some(tuple) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
            // Convert the tuple contents to CPython
            let cpython_tuple =
                pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
                    let items: Vec<pyo3::Bound<'_, pyo3::PyAny>> = tuple
                        .iter()
                        .map(|item| {
                            let converted = to_pyo3_object(item, vm).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                    "Failed to convert tuple item: {:?}",
                                    e
                                ))
                            })?;
                            converted.to_pyo3(py)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(pyo3::types::PyTuple::new(py, &items)?.into_any().unbind())
                })
                .map_err(|e| pyo3_err_to_rustpython(e, vm))?;
            return Ok(ToPyo3Ref::OwnedNative(cpython_tuple));
        }

        // Check if it's a list - convert each item recursively
        if let Some(list) = obj.downcast_ref::<rustpython_vm::builtins::PyList>() {
            let cpython_list = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
                let items: Vec<pyo3::Bound<'_, pyo3::PyAny>> = list
                    .borrow_vec()
                    .iter()
                    .map(|item| {
                        let converted = to_pyo3_object(item, vm).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to convert list item: {:?}",
                                e
                            ))
                        })?;
                        converted.to_pyo3(py)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(pyo3::types::PyList::new(py, &items)?.into_any().unbind())
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm))?;
            return Ok(ToPyo3Ref::OwnedNative(cpython_list));
        }

        // Check if it supports buffer protocol - create shared memoryview for zero-copy
        // This enables ctypes.from_buffer() and similar APIs to share memory
        // Note: Exclude bytes/bytearray - they should be passed as-is via pickle
        //       because ctypes expects bytes, not memoryview (e.g., c_char(b'x'))
        if !obj.class().is(vm.ctx.types.bytes_type)
            && !obj.class().is(vm.ctx.types.bytearray_type)
            && let Ok(buffer) = PyBuffer::try_from_borrowed_object(vm, obj)
        {
            return create_shared_buffer_view(obj, buffer, vm);
        }

        // Check if it's a callable (function, method, lambda, etc.)
        // We need to handle callables specially because pickle can't serialize
        // user-defined functions properly (they fail on unpickle with
        // "Can't get attribute 'func_name' on <module '__main__' ...>")
        // Note: We exclude types/classes since they should be pickled normally,
        // only wrap actual function/method/lambda callables
        if obj.is_callable() && obj.downcast_ref::<PyType>().is_none() {
            return create_rustpython_callback_wrapper(obj, vm);
        }

        // Default: try to pickle
        let pickled = rustpython_pickle_dumps(obj.to_owned(), vm)?;
        Ok(ToPyo3Ref::Pickled(pickled))
    }

    /// Create a CPython callable wrapper for a RustPython callable.
    /// This allows RustPython functions to be passed as callbacks to CPython code.
    /// The wrapper implements the descriptor protocol (__get__) so it works correctly
    /// as a method when stored in a class __dict__.
    fn create_rustpython_callback_wrapper(
        callable: &PyObject,
        vm: &VirtualMachine,
    ) -> PyResult<ToPyo3Ref<'static>> {
        // Store the callable in an Arc so it can be moved into the closure
        let callable_arc: Arc<PyObjectRef> = Arc::new(callable.to_owned());

        let cpython_wrapper = pyo3::Python::attach(|py| -> Result<pyo3::Py<pyo3::PyAny>, PyErr> {
            // Create a closure that captures the RustPython callable
            let callable_clone = Arc::clone(&callable_arc);
            let wrapper_fn = move |args: &pyo3::Bound<'_, pyo3::types::PyTuple>,
                                   _kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>|
                  -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                // Access the current RustPython VM via thread-local storage
                // The VM should be set when RustPython code is executing
                // Callback exceptions are "unraisable" - they go through sys.unraisablehook
                let result: Option<pyo3::Py<pyo3::PyAny>> =
                    rustpython_vm::vm::thread::with_current_vm(|vm| {
                        // Helper to handle exceptions as unraisable
                        let handle_exception =
                            |e: rustpython_vm::builtins::PyBaseExceptionRef,
                             callable: &PyObjectRef,
                             vm: &VirtualMachine| {
                                // Format the message like CPython
                                let callable_repr = callable
                                    .repr(vm)
                                    .map(|s| s.as_str().to_owned())
                                    .unwrap_or_else(|_| "<unknown>".to_owned());
                                let msg = format!(
                                    "Exception ignored on calling ctypes callback function {}",
                                    callable_repr
                                );
                                vm.run_unraisable(e, Some(msg), vm.ctx.none());
                            };

                        // Convert CPython args to RustPython
                        let rp_args: Vec<PyObjectRef> = match args
                            .iter()
                            .map(|arg| {
                                let pyo3_ref = create_pyo3_object(args.py(), &arg);
                                pyo3_to_rustpython(pyo3_ref, vm)
                            })
                            .collect::<Result<Vec<_>, _>>()
                        {
                            Ok(args) => args,
                            Err(e) => {
                                handle_exception(e, &callable_clone, vm);
                                return None;
                            }
                        };

                        // Call the RustPython callable
                        let call_result = match callable_clone.call(rp_args, vm) {
                            Ok(r) => r,
                            Err(e) => {
                                handle_exception(e, &callable_clone, vm);
                                return None;
                            }
                        };

                        // Convert RustPython result back to CPython
                        let pyo3_result = match to_pyo3_object(&call_result, vm) {
                            Ok(r) => r,
                            Err(e) => {
                                handle_exception(e, &callable_clone, vm);
                                return None;
                            }
                        };

                        match pyo3::Python::attach(|py| {
                            pyo3_result.to_pyo3(py).map(|bound| bound.unbind())
                        }) {
                            Ok(r) => Some(r),
                            Err(e) => {
                                // CPython error - convert and handle as unraisable
                                let rp_err = pyo3_err_to_rustpython(e, vm);
                                handle_exception(rp_err, &callable_clone, vm);
                                None
                            }
                        }
                    });

                // Return None as default value when exception occurred
                Ok(result.unwrap_or_else(|| pyo3::Python::attach(|py| py.None())))
            };

            // Create the CPython function from the closure
            let py_func = PyCFunction::new_closure(py, None, None, wrapper_fn)?;

            // Wrap in a descriptor that implements __get__ for proper method binding.
            // This is necessary because PyCFunction doesn't bind like a Python function
            // when accessed through a class __dict__.
            // We create a simple Python class with __call__ and __get__ methods.
            let wrapper_class_code = r#"
class _MethodDescriptor:
    __slots__ = ('_func',)
    def __init__(self, func):
        self._func = func
    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        import types
        return types.MethodType(self._func, obj)
_MethodDescriptor
"#;
            let builtins = py.import("builtins")?;
            let exec_fn = builtins.getattr("exec")?;
            let globals = pyo3::types::PyDict::new(py);
            exec_fn.call1((wrapper_class_code, &globals))?;
            let wrapper_class = globals.get_item("_MethodDescriptor")?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to create _MethodDescriptor",
                )
            })?;
            let wrapper_instance = wrapper_class.call1((py_func,))?;
            Ok(wrapper_instance.unbind())
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        Ok(ToPyo3Ref::OwnedNative(cpython_wrapper))
    }

    /// Execute binary operation on CPython objects
    fn pyo3_binary_op(a: &PyObject, b: &PyObject, op: &str, vm: &VirtualMachine) -> PyResult {
        // If neither is Pyo3Ref, return NotImplemented
        if a.downcast_ref::<Pyo3Ref>().is_none() && b.downcast_ref::<Pyo3Ref>().is_none() {
            return Ok(vm.ctx.not_implemented());
        }

        let a_obj = to_pyo3_object(a, vm)?;
        let b_obj = to_pyo3_object(b, vm)?;

        let result = pyo3::Python::attach(|py| -> Result<PyArithmeticValue<Pyo3Ref>, PyErr> {
            let a_py = a_obj.to_pyo3(py)?;
            let b_py = b_obj.to_pyo3(py)?;

            let result_obj = a_py.call_method1(op, (&b_py,))?;

            if result_obj.is(py.NotImplemented()) {
                return Ok(PyArithmeticValue::NotImplemented);
            }

            Ok(PyArithmeticValue::Implemented(create_pyo3_object(
                py,
                &result_obj,
            )))
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        match result {
            PyArithmeticValue::NotImplemented => Ok(vm.ctx.not_implemented()),
            PyArithmeticValue::Implemented(pyo3_obj) => pyo3_to_rustpython(pyo3_obj, vm),
        }
    }

    /// Execute binary operation on CPython object with direct pyo3 object reference.
    /// Used by Pyo3Type.__mul__ where we already have the pyo3 object.
    fn pyo3_binary_op_with_pyo3ref(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        other: &PyObject,
        op: &str,
        vm: &VirtualMachine,
    ) -> PyResult {
        let other_obj = to_pyo3_object(other, vm)?;

        let result = pyo3::Python::attach(|py| -> Result<PyArithmeticValue<Pyo3Ref>, PyErr> {
            let a_py = py_obj.bind(py);
            let b_py = other_obj.to_pyo3(py)?;

            let result_obj = a_py.call_method1(op, (&b_py,))?;

            if result_obj.is(py.NotImplemented()) {
                return Ok(PyArithmeticValue::NotImplemented);
            }

            Ok(PyArithmeticValue::Implemented(create_pyo3_object(
                py,
                &result_obj,
            )))
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

        match result {
            PyArithmeticValue::NotImplemented => Ok(vm.ctx.not_implemented()),
            PyArithmeticValue::Implemented(pyo3_obj) => pyo3_to_rustpython(pyo3_obj, vm),
        }
    }

    /// Wrapper for CPython objects
    #[pyattr]
    #[pyclass(name = "ref")]
    #[derive(PyPayload)]
    struct Pyo3Ref {
        py_obj: pyo3::Py<pyo3::PyAny>,
        /// Pickled bytes for potential unpickling to native RustPython object
        pickled: Option<Vec<u8>>,
    }

    impl std::fmt::Debug for Pyo3Ref {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Pyo3Ref")
                .field("py_obj", &"<CPython object>")
                .finish()
        }
    }

    impl GetAttr for Pyo3Ref {
        fn getattro(zelf: &Py<Self>, name: &Py<PyStr>, vm: &VirtualMachine) -> PyResult {
            pyo3_getattr_impl(&zelf.py_obj, name.as_str(), vm)
        }
    }

    impl SetAttr for Pyo3Ref {
        fn setattro(
            zelf: &Py<Self>,
            attr_name: &Py<PyStr>,
            value: PySetterValue,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            pyo3::Python::attach(|py| -> Result<(), pyo3::PyErr> {
                let obj = zelf.py_obj.bind(py);
                match value {
                    PySetterValue::Assign(val) => {
                        let pyo3_val = to_pyo3_object(&val, vm).map_err(|e| {
                            pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to convert value: {:?}",
                                e
                            ))
                        })?;
                        let pyo3_bound = pyo3_val.to_pyo3(py)?;
                        obj.setattr(attr_name.as_str(), pyo3_bound)?;
                    }
                    PySetterValue::Delete => {
                        obj.delattr(attr_name.as_str())?;
                    }
                }
                Ok(())
            })
            .map_err(|e| vm.new_attribute_error(format!("CPython setattr error: {}", e)))
        }
    }

    impl Callable for Pyo3Ref {
        type Args = FuncArgs;

        fn call(zelf: &Py<Self>, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
            pyo3_call_impl(&zelf.py_obj, args, vm)
        }
    }

    impl Representable for Pyo3Ref {
        fn repr_str(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<String> {
            // Get repr from CPython directly
            let result = pyo3::Python::attach(|py| -> Result<String, PyErr> {
                let obj = zelf.py_obj.bind(py);
                let builtins = py.import("builtins")?;
                let repr_fn = builtins.getattr("repr")?;
                let repr_result = repr_fn.call1((obj,))?;
                repr_result.extract()
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm))?;
            Ok(result)
        }
    }

    impl GetDescriptor for Pyo3Ref {
        fn descr_get(
            zelf: PyObjectRef,
            obj: Option<PyObjectRef>,
            cls: Option<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult {
            let (zelf_ref, obj) = Self::_unwrap(&zelf, obj, vm)?;

            // obj가 None이면 unbound descriptor로 반환
            if vm.is_none(&obj) {
                return Ok(zelf);
            }

            // CPython 객체의 __get__ 호출
            pyo3::Python::attach(|py| -> Result<PyObjectRef, pyo3::PyErr> {
                let pyo3_obj = zelf_ref.py_obj.bind(py);

                // __get__ 메서드가 있는지 확인
                if let Ok(get_method) = pyo3_obj.getattr("__get__") {
                    // obj를 CPython 객체로 변환
                    let pyo3_instance = to_pyo3_object(&obj, vm).map_err(|e| {
                        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to convert obj: {:?}",
                            e
                        ))
                    })?;
                    let pyo3_instance_bound = pyo3_instance.to_pyo3(py)?;

                    // cls 변환 (있으면)
                    let pyo3_cls = if let Some(ref c) = cls {
                        let c_conv = to_pyo3_object(c, vm).map_err(|e| {
                            pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to convert cls: {:?}",
                                e
                            ))
                        })?;
                        c_conv.to_pyo3(py)?
                    } else {
                        py.None().into_bound(py)
                    };

                    // __get__(obj, type) 호출
                    let result = get_method.call1((pyo3_instance_bound, pyo3_cls))?;
                    let result_ref = create_pyo3_object(py, &result);
                    pyo3_to_rustpython(result_ref, vm).map_err(|e| {
                        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to convert result: {:?}",
                            e
                        ))
                    })
                } else {
                    // __get__이 없으면 그냥 self 반환 (non-descriptor)
                    Ok(zelf.clone())
                }
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm))
        }
    }

    impl AsNumber for Pyo3Ref {
        fn as_number() -> &'static PyNumberMethods {
            static AS_NUMBER: PyNumberMethods = PyNumberMethods {
                add: Some(|a, b, vm| pyo3_binary_op(a, b, "__add__", vm)),
                subtract: Some(|a, b, vm| pyo3_binary_op(a, b, "__sub__", vm)),
                multiply: Some(|a, b, vm| pyo3_binary_op(a, b, "__mul__", vm)),
                remainder: Some(|a, b, vm| pyo3_binary_op(a, b, "__mod__", vm)),
                divmod: Some(|a, b, vm| pyo3_binary_op(a, b, "__divmod__", vm)),
                floor_divide: Some(|a, b, vm| pyo3_binary_op(a, b, "__floordiv__", vm)),
                true_divide: Some(|a, b, vm| pyo3_binary_op(a, b, "__truediv__", vm)),
                ..PyNumberMethods::NOT_IMPLEMENTED
            };
            &AS_NUMBER
        }
    }

    impl Comparable for Pyo3Ref {
        fn cmp(
            zelf: &Py<Self>,
            other: &PyObject,
            op: PyComparisonOp,
            vm: &VirtualMachine,
        ) -> PyResult<PyComparisonValue> {
            let method_name = match op {
                PyComparisonOp::Lt => "__lt__",
                PyComparisonOp::Le => "__le__",
                PyComparisonOp::Eq => "__eq__",
                PyComparisonOp::Ne => "__ne__",
                PyComparisonOp::Gt => "__gt__",
                PyComparisonOp::Ge => "__ge__",
            };

            let other_obj = to_pyo3_object(other, vm)?;

            let result = pyo3::Python::attach(|py| -> Result<PyComparisonValue, PyErr> {
                let obj = zelf.py_obj.bind(py);
                let other_py = other_obj.to_pyo3(py)?;

                let result = obj.call_method1(method_name, (&other_py,))?;

                if result.is(py.NotImplemented()) {
                    return Ok(PyComparisonValue::NotImplemented);
                }

                // Try to extract bool; if it fails, return NotImplemented
                match result.extract::<bool>() {
                    Ok(bool_val) => Ok(PyComparisonValue::Implemented(bool_val)),
                    Err(_) => Ok(PyComparisonValue::NotImplemented),
                }
            })
            .map_err(|e| pyo3_err_to_rustpython(e, vm))?;

            Ok(result)
        }
    }

    /// Helper to get len from CPython object
    fn pyo3_len(py_obj: &pyo3::Py<pyo3::PyAny>, vm: &VirtualMachine) -> PyResult<usize> {
        pyo3::Python::attach(|py| -> Result<usize, PyErr> {
            let obj = py_obj.bind(py);
            let builtins = py.import("builtins")?;
            let len_fn = builtins.getattr("len")?;
            let len_result = len_fn.call1((obj,))?;
            len_result.extract()
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))
    }

    /// Helper to get item by index from CPython object
    fn pyo3_getitem_by_index(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        index: isize,
        vm: &VirtualMachine,
    ) -> PyResult {
        let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let obj = py_obj.bind(py);
            let item = obj.get_item(index)?;
            Ok(create_pyo3_object(py, &item))
        })
        .map_err(|e| vm.new_index_error(format!("CPython getitem error: {}", e)))?;

        pyo3_to_rustpython(pyo3_obj, vm)
    }

    /// Helper to get item by key from CPython object
    fn pyo3_getitem(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        key: &PyObject,
        vm: &VirtualMachine,
    ) -> PyResult {
        let key_obj = to_pyo3_object(key, vm)?;

        let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let obj = py_obj.bind(py);
            let key_py = key_obj.to_pyo3(py)?;
            let item = obj.get_item(&key_py)?;
            Ok(create_pyo3_object(py, &item))
        })
        .map_err(|e| {
            vm.new_key_error(
                vm.ctx
                    .new_str(format!("CPython getitem error: {}", e))
                    .into(),
            )
        })?;

        pyo3_to_rustpython(pyo3_obj, vm)
    }

    /// Helper to set item in CPython object
    fn pyo3_setitem(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        key: &PyObject,
        value: Option<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let key_obj = to_pyo3_object(key, vm)?;
        let value_obj = value.as_ref().map(|v| to_pyo3_object(v, vm)).transpose()?;

        pyo3::Python::attach(|py| -> Result<(), PyErr> {
            let obj = py_obj.bind(py);
            let key_py = key_obj.to_pyo3(py)?;

            match value_obj {
                Some(ref val_obj) => {
                    let val_py = val_obj.to_pyo3(py)?;
                    obj.set_item(&key_py, &val_py)?;
                }
                None => {
                    obj.del_item(&key_py)?;
                }
            }
            Ok(())
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))
    }

    /// Helper to check if item is in CPython object
    fn pyo3_contains(
        py_obj: &pyo3::Py<pyo3::PyAny>,
        target: &PyObject,
        vm: &VirtualMachine,
    ) -> PyResult<bool> {
        let target_obj = to_pyo3_object(target, vm)?;

        pyo3::Python::attach(|py| -> Result<bool, PyErr> {
            let obj = py_obj.bind(py);
            let target_py = target_obj.to_pyo3(py)?;
            obj.contains(&target_py)
        })
        .map_err(|e| pyo3_err_to_rustpython(e, vm))
    }

    impl AsSequence for Pyo3Ref {
        fn as_sequence() -> &'static PySequenceMethods {
            static AS_SEQUENCE: PySequenceMethods = PySequenceMethods {
                length: AtomicCell::new(Some(|seq, vm| {
                    let zelf = Pyo3Ref::sequence_downcast(seq);
                    pyo3_len(&zelf.py_obj, vm)
                })),
                concat: AtomicCell::new(None),
                repeat: AtomicCell::new(None),
                item: AtomicCell::new(Some(|seq, i, vm| {
                    let zelf = Pyo3Ref::sequence_downcast(seq);
                    pyo3_getitem_by_index(&zelf.py_obj, i, vm)
                })),
                ass_item: AtomicCell::new(None),
                contains: AtomicCell::new(Some(|seq, target, vm| {
                    let zelf = Pyo3Ref::sequence_downcast(seq);
                    pyo3_contains(&zelf.py_obj, target, vm)
                })),
                inplace_concat: AtomicCell::new(None),
                inplace_repeat: AtomicCell::new(None),
            };
            &AS_SEQUENCE
        }
    }

    impl AsMapping for Pyo3Ref {
        fn as_mapping() -> &'static PyMappingMethods {
            static AS_MAPPING: PyMappingMethods = PyMappingMethods {
                length: AtomicCell::new(Some(|mapping, vm| {
                    let zelf = Pyo3Ref::mapping_downcast(mapping);
                    pyo3_len(&zelf.py_obj, vm)
                })),
                subscript: AtomicCell::new(Some(|mapping, needle, vm| {
                    let zelf = Pyo3Ref::mapping_downcast(mapping);
                    pyo3_getitem(&zelf.py_obj, needle, vm)
                })),
                ass_subscript: AtomicCell::new(Some(|mapping, needle, value, vm| {
                    let zelf = Pyo3Ref::mapping_downcast(mapping);
                    pyo3_setitem(&zelf.py_obj, needle, value, vm)
                })),
            };
            &AS_MAPPING
        }
    }

    impl Iterable for Pyo3Ref {
        fn iter(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult {
            let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
                let obj = zelf.py_obj.bind(py);
                let builtins = py.import("builtins")?;
                let iter_fn = builtins.getattr("iter")?;
                let iter_result = iter_fn.call1((obj,))?;
                Ok(create_pyo3_object(py, &iter_result))
            })
            .map_err(|e| vm.new_type_error(format!("CPython iter error: {}", e)))?;

            // Iterators should stay as Pyo3Ref, don't try to unpickle
            Ok(pyo3_obj.into_ref(&vm.ctx).into())
        }
    }

    use rustpython_vm::types::IterNext;

    #[pyclass(with(
        GetAttr,
        SetAttr,
        Callable,
        Representable,
        GetDescriptor,
        AsNumber,
        Comparable,
        AsSequence,
        AsMapping,
        Iterable,
        IterNext
    ))]
    impl Pyo3Ref {}

    impl IterNext for Pyo3Ref {
        fn next(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyIterReturn> {
            let result = pyo3::Python::attach(|py| -> Result<Option<Pyo3Ref>, pyo3::PyErr> {
                let obj = zelf.py_obj.bind(py);
                let builtins = py.import("builtins")?;
                let next_fn = builtins.getattr("next")?;

                match next_fn.call1((obj,)) {
                    Ok(result) => Ok(Some(create_pyo3_object(py, &result))),
                    Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) => Ok(None),
                    Err(e) => Err(e),
                }
            });

            match result {
                Ok(Some(pyo3_obj)) => {
                    let result = pyo3_to_rustpython(pyo3_obj, vm)?;
                    Ok(PyIterReturn::Return(result))
                }
                Ok(None) => Ok(PyIterReturn::StopIteration(None)),
                Err(e) => Err(pyo3_err_to_rustpython(e, vm)),
            }
        }
    }

    /// Import a module from CPython and return a wrapper object.
    ///
    /// # Arguments
    /// * `name` - The name of the module to import
    ///
    /// # Returns
    /// A Pyo3Ref wrapping the imported module
    #[pyfunction]
    fn import_module(name: PyStrRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        import_module_impl(name.as_str(), vm)
    }

    /// Internal implementation for importing a module from CPython.
    /// Used by both the Python API and borrow_module.
    pub(crate) fn import_module_impl(
        module_name: &str,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let module_name_owned = module_name.to_owned();

        let pyo3_obj = pyo3::Python::attach(|py| -> Result<Pyo3Ref, PyErr> {
            let module = py.import(&*module_name_owned)?;
            Ok(create_pyo3_object(py, module.as_any()))
        })
        .map_err(|e| {
            vm.new_import_error(
                format!("Cannot import '{}': {}", module_name, e),
                vm.ctx.new_str(module_name),
            )
        })?;

        // Modules should stay as Pyo3Ref, don't try to unpickle
        Ok(pyo3_obj.into_ref(&vm.ctx).into())
    }
}
