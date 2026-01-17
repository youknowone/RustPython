use rustpython_vm::{InterpreterBuilder, VirtualMachine};

/// Extension trait for InterpreterBuilder to add rustpython-specific functionality.
pub trait InterpreterBuilderExt {
    /// Initialize the Python standard library.
    ///
    /// Requires the `stdlib` feature to be enabled.
    #[cfg(feature = "stdlib")]
    fn init_stdlib(self) -> Self;
}

impl InterpreterBuilderExt for InterpreterBuilder {
    #[cfg(feature = "stdlib")]
    fn init_stdlib(self) -> Self {
        let defs = rustpython_stdlib::stdlib_module_defs(&self.ctx);
        self.add_native_modules(&defs).init_hook(|vm| {
            #[cfg(feature = "freeze-stdlib")]
            setup_frozen_stdlib(vm);

            #[cfg(not(feature = "freeze-stdlib"))]
            setup_dynamic_stdlib(vm);
        })
    }
}

/// Setup frozen standard library (compiled into the binary)
#[cfg(all(feature = "stdlib", feature = "freeze-stdlib"))]
fn setup_frozen_stdlib(vm: &mut VirtualMachine) {
    vm.add_frozen(rustpython_pylib::FROZEN_STDLIB);

    // FIXME: Remove this hack once sys._stdlib_dir is properly implemented
    // or _frozen_importlib doesn't depend on it anymore.
    assert!(vm.sys_module.get_attr("_stdlib_dir", vm).is_err());
    vm.sys_module
        .set_attr(
            "_stdlib_dir",
            vm.new_pyobj(rustpython_pylib::LIB_PATH.to_owned()),
            vm,
        )
        .unwrap();
}

/// Setup dynamic standard library loading from filesystem
#[cfg(all(feature = "stdlib", not(feature = "freeze-stdlib")))]
fn setup_dynamic_stdlib(vm: &mut VirtualMachine) {
    use rustpython_vm::common::rc::PyRc;

    let state = PyRc::get_mut(&mut vm.state).unwrap();
    let paths = collect_stdlib_paths();

    // Insert at the beginning so stdlib comes before user paths
    for path in paths.into_iter().rev() {
        state.config.paths.module_search_paths.insert(0, path);
    }
}

/// Collect standard library paths from build-time configuration
#[cfg(all(feature = "stdlib", not(feature = "freeze-stdlib")))]
fn collect_stdlib_paths() -> Vec<String> {
    // BUILDTIME_RUSTPYTHONPATH should be set when distributing
    if let Some(paths) = option_env!("BUILDTIME_RUSTPYTHONPATH") {
        crate::settings::split_paths(paths)
            .map(|path| path.into_os_string().into_string().unwrap())
            .collect()
    } else {
        #[cfg(feature = "rustpython-pylib")]
        {
            vec![rustpython_pylib::LIB_PATH.to_owned()]
        }
        #[cfg(not(feature = "rustpython-pylib"))]
        {
            vec![]
        }
    }
}
