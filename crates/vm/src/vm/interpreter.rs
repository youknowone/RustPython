use super::{Context, PyConfig, VirtualMachine, setting::Settings, thread};
use crate::{
    PyResult, builtins, common::rc::PyRc, getpath, stdlib::atexit, vm::PyBaseExceptionRef,
};
use core::sync::atomic::Ordering;

type InitFunc = Box<dyn FnOnce(&mut VirtualMachine)>;

/// Configuration builder for constructing an Interpreter.
///
/// This is the preferred way to configure and create an interpreter with custom modules.
/// Modules must be registered before the interpreter is built,
/// similar to CPython's `PyImport_AppendInittab` which must be called before `Py_Initialize`.
///
/// # Example
/// ```
/// use rustpython_vm::Interpreter;
///
/// let interp = Interpreter::builder()
///     .settings(Default::default())
///     .add_modules(rustpython_stdlib::stdlib_module_defs())
///     .build();
/// ```
pub struct InterpreterBuilder {
    settings: Settings,
    pub ctx: PyRc<Context>,
    module_defs: Vec<&'static builtins::PyModuleDef>,
    init_hooks: Vec<InitFunc>,
}

impl InterpreterBuilder {
    /// Create a new interpreter configuration with default settings.
    pub fn new() -> Self {
        Self {
            settings: Settings::default(),
            ctx: Context::genesis().clone(),
            module_defs: Vec::new(),
            init_hooks: Vec::new(),
        }
    }

    /// Set custom settings for the interpreter.
    ///
    /// If called multiple times, only the last settings will be used.
    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    /// Add a single native module definition.
    ///
    /// # Example
    /// ```
    /// use rustpython_vm::Interpreter;
    ///
    /// let interp = Interpreter::builder()
    ///     .init_hook(|vm| {
    ///         let def = mymodule::module_def(&vm.ctx);
    ///         rustpython_vm::common::rc::PyRc::get_mut(&mut vm.state)
    ///             .unwrap()
    ///             .module_defs
    ///             .insert(def.name.as_str(), def);
    ///     })
    ///     .build();
    /// ```
    pub fn add_native_module(self, def: &'static builtins::PyModuleDef) -> Self {
        self.add_native_modules(&[def])
    }

    /// Add multiple native module definitions.
    ///
    /// # Example
    /// ```
    /// use rustpython_vm::Interpreter;
    ///
    /// let builder = Interpreter::builder();
    /// let defs = rustpython_stdlib::stdlib_module_defs(&builder.ctx);
    /// let interp = builder.add_native_modules(&defs).build();
    /// ```
    pub fn add_native_modules(mut self, defs: &[&'static builtins::PyModuleDef]) -> Self {
        self.module_defs.extend_from_slice(defs);
        self
    }

    /// Add a custom initialization hook.
    ///
    /// Hooks are executed in the order they are added during interpreter creation.
    /// This function will be called after modules are registered but before
    /// the VM is initialized, allowing for additional customization.
    ///
    /// # Example
    /// ```
    /// use rustpython_vm::Interpreter;
    ///
    /// let interp = Interpreter::builder()
    ///     .init_hook(|vm| {
    ///         // Custom initialization
    ///     })
    ///     .build();
    /// ```
    pub fn init_hook<F>(mut self, init: F) -> Self
    where
        F: FnOnce(&mut VirtualMachine) + 'static,
    {
        self.init_hooks.push(Box::new(init));
        self
    }

    /// Build the interpreter.
    ///
    /// This consumes the configuration and returns a fully initialized Interpreter.
    pub fn build(self) -> Interpreter {
        let paths = getpath::init_path_config(&self.settings);
        let config = PyConfig::new(self.settings, paths);

        let ctx = self.ctx;
        crate::types::TypeZoo::extend(&ctx);
        crate::exceptions::ExceptionZoo::extend(&ctx);

        let mut vm = VirtualMachine::new(config, ctx);

        // Add custom module definitions
        for def in self.module_defs {
            PyRc::get_mut(&mut vm.state)
                .expect("there should not be multiple threads during initialization")
                .module_defs
                .insert(def.name.as_str(), def);
        }

        // Call init hooks in order
        for hook in self.init_hooks {
            hook(&mut vm);
        }

        vm.initialize();

        Interpreter { vm }
    }

    /// Alias for `build()` for compatibility with the `interpreter()` pattern.
    pub fn interpreter(self) -> Interpreter {
        self.build()
    }
}

impl Default for InterpreterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The general interface for the VM
///
/// # Examples
/// Runs a simple embedded hello world program.
/// ```
/// use rustpython_vm::Interpreter;
/// use rustpython_vm::compiler::Mode;
/// Interpreter::without_stdlib(Default::default()).enter(|vm| {
///     let scope = vm.new_scope_with_builtins();
///     let source = r#"print("Hello World!")"#;
///     let code_obj = vm.compile(
///             source,
///             Mode::Exec,
///             "<embedded>".to_owned(),
///     ).map_err(|err| vm.new_syntax_error(&err, Some(source))).unwrap();
///     vm.run_code_obj(code_obj, scope).unwrap();
/// });
/// ```
pub struct Interpreter {
    vm: VirtualMachine,
}

impl Interpreter {
    /// Create a new interpreter configuration builder.
    ///
    /// # Example
    /// ```
    /// use rustpython_vm::Interpreter;
    ///
    /// let interp = Interpreter::builder(Default::default())
    ///     .add_native_modules(rustpython_stdlib::stdlib_module_defs())
    ///     .build();
    /// ```
    pub fn builder(settings: Settings) -> InterpreterBuilder {
        InterpreterBuilder::new().settings(settings)
    }

    /// This is a bare unit to build up an interpreter without the standard library.
    /// To create an interpreter with the standard library with the `rustpython` crate, use `rustpython::InterpreterBuilder`.
    /// To create an interpreter without the `rustpython` crate, but only with `rustpython-vm`,
    /// try to build one from the source code of `InterpreterBuilder`. It will not be a one-liner but it also will not be too hard.
    pub fn without_stdlib(settings: Settings) -> Self {
        Self::with_init(settings, |_| {})
    }

    /// Create with initialize function taking mutable vm reference.
    /// ```
    /// use rustpython_vm::Interpreter;
    /// Interpreter::with_init(Default::default(), |vm| {
    ///     // put this line to add stdlib to the vm
    ///     // vm.add_native_module_defs(rustpython_stdlib::stdlib_module_defs());
    /// }).enter(|vm| {
    ///     vm.run_code_string(vm.new_scope_with_builtins(), "print(1)", "<...>".to_owned());
    /// });
    /// ```
    pub fn with_init<F>(settings: Settings, init: F) -> Self
    where
        F: FnOnce(&mut VirtualMachine),
    {
        // Compute path configuration from settings
        let paths = getpath::init_path_config(&settings);
        let config = PyConfig::new(settings, paths);

        let ctx = Context::genesis();
        crate::types::TypeZoo::extend(ctx);
        crate::exceptions::ExceptionZoo::extend(ctx);
        let mut vm = VirtualMachine::new(config, ctx.clone());
        init(&mut vm);
        vm.initialize();
        Self { vm }
    }

    /// Run a function with the main virtual machine and return a PyResult of the result.
    ///
    /// To enter vm context multiple times or to avoid buffer/exception management, this function is preferred.
    /// `enter` is lightweight and it returns a python object in PyResult.
    /// You can stop or continue the execution multiple times by calling `enter`.
    ///
    /// To finalize the vm once all desired `enter`s are called, calling `finalize` will be helpful.
    ///
    /// See also [`Interpreter::run`] for managed way to run the interpreter.
    pub fn enter<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&VirtualMachine) -> R,
    {
        thread::enter_vm(&self.vm, || f(&self.vm))
    }

    /// Run [`Interpreter::enter`] and call [`VirtualMachine::expect_pyresult`] for the result.
    ///
    /// This function is useful when you want to expect a result from the function,
    /// but also print useful panic information when exception raised.
    ///
    /// See also [`Interpreter::enter`] and [`VirtualMachine::expect_pyresult`] for more information.
    pub fn enter_and_expect<F, R>(&self, f: F, msg: &str) -> R
    where
        F: FnOnce(&VirtualMachine) -> PyResult<R>,
    {
        self.enter(|vm| {
            let result = f(vm);
            vm.expect_pyresult(result, msg)
        })
    }

    /// Run a function with the main virtual machine and return exit code.
    ///
    /// To enter vm context only once and safely terminate the vm, this function is preferred.
    /// Unlike [`Interpreter::enter`], `run` calls finalize and returns exit code.
    /// You will not be able to obtain Python exception in this way.
    ///
    /// See [`Interpreter::finalize`] for the finalization steps.
    /// See also [`Interpreter::enter`] for pure function call to obtain Python exception.
    pub fn run<F>(self, f: F) -> u32
    where
        F: FnOnce(&VirtualMachine) -> PyResult<()>,
    {
        let res = self.enter(|vm| f(vm));
        self.finalize(res.err())
    }

    /// Finalize vm and turns an exception to exit code.
    ///
    /// Finalization steps (matching Py_FinalizeEx):
    /// 1. Flush stdout and stderr.
    /// 1. Handle exit exception and turn it to exit code.
    /// 1. Wait for thread shutdown (call threading._shutdown).
    /// 1. Mark vm as finalizing.
    /// 1. Run atexit exit functions.
    /// 1. Mark vm as finalized.
    ///
    /// Note that calling `finalize` is not necessary by purpose though.
    pub fn finalize(self, exc: Option<PyBaseExceptionRef>) -> u32 {
        self.enter(|vm| {
            vm.flush_std();

            // See if any exception leaked out:
            let exit_code = if let Some(exc) = exc {
                vm.handle_exit_exception(exc)
            } else {
                0
            };

            // Wait for thread shutdown - call threading._shutdown() if available.
            // This waits for all non-daemon threads to complete.
            // threading module may not be imported, so ignore import errors.
            if let Ok(threading) = vm.import("threading", 0)
                && let Ok(shutdown) = threading.get_attr("_shutdown", vm)
                && let Err(e) = shutdown.call((), vm)
            {
                vm.run_unraisable(
                    e,
                    Some("Exception ignored in threading shutdown".to_owned()),
                    threading,
                );
            }

            // Mark as finalizing AFTER thread shutdown
            vm.state.finalizing.store(true, Ordering::Release);

            // Run atexit exit functions
            atexit::_run_exitfuncs(vm);

            vm.flush_std();

            exit_code
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PyObjectRef,
        builtins::{PyStr, int},
    };
    use malachite_bigint::ToBigInt;

    #[test]
    fn test_add_py_integers() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let a: PyObjectRef = vm.ctx.new_int(33_i32).into();
            let b: PyObjectRef = vm.ctx.new_int(12_i32).into();
            let res = vm._add(&a, &b).unwrap();
            let value = int::get_value(&res);
            assert_eq!(*value, 45_i32.to_bigint().unwrap());
        })
    }

    #[test]
    fn test_multiply_str() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let a = vm.new_pyobj(crate::common::ascii!("Hello "));
            let b = vm.new_pyobj(4_i32);
            let res = vm._mul(&a, &b).unwrap();
            let value = res.downcast_ref::<PyStr>().unwrap();
            assert_eq!(value.as_str(), "Hello Hello Hello Hello ")
        })
    }
}
