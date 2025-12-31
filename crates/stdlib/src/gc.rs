pub(crate) use gc::make_module;

#[pymodule]
mod gc {
    use crate::vm::{
        PyObjectRef, PyResult, VirtualMachine,
        builtins::PyListRef,
        function::{FromArgs, FuncArgs, OptionalArg},
        gc_state,
    };

    // Debug flag constants
    #[pyattr]
    const DEBUG_STATS: u32 = gc_state::DEBUG_STATS;
    #[pyattr]
    const DEBUG_COLLECTABLE: u32 = gc_state::DEBUG_COLLECTABLE;
    #[pyattr]
    const DEBUG_UNCOLLECTABLE: u32 = gc_state::DEBUG_UNCOLLECTABLE;
    #[pyattr]
    const DEBUG_SAVEALL: u32 = gc_state::DEBUG_SAVEALL;
    #[pyattr]
    const DEBUG_LEAK: u32 = gc_state::DEBUG_LEAK;

    /// Enable automatic garbage collection.
    #[pyfunction]
    fn enable() {
        gc_state::gc_state().enable();
    }

    /// Disable automatic garbage collection.
    #[pyfunction]
    fn disable() {
        gc_state::gc_state().disable();
    }

    /// Return true if automatic gc is enabled.
    #[pyfunction]
    fn isenabled() -> bool {
        gc_state::gc_state().is_enabled()
    }

    /// Run a garbage collection. Returns the number of unreachable objects found.
    #[derive(FromArgs)]
    struct CollectArgs {
        #[pyarg(any, optional)]
        generation: OptionalArg<i32>,
    }

    #[pyfunction]
    fn collect(args: CollectArgs, vm: &VirtualMachine) -> PyResult<i32> {
        let generation = args.generation;
        let generation_num = generation.unwrap_or(2);
        if generation_num < 0 || generation_num > 2 {
            return Err(vm.new_value_error("invalid generation".to_owned()));
        }

        // Invoke callbacks with "start" phase
        invoke_callbacks(vm, "start", generation_num as usize);

        // Manual gc.collect() should run even if GC is disabled
        let (collected, _uncollectable) =
            gc_state::gc_state().collect_force(generation_num as usize);

        // Invoke callbacks with "stop" phase
        invoke_callbacks(vm, "stop", generation_num as usize);

        Ok(collected as i32)
    }

    /// Return the current collection thresholds as a tuple.
    #[pyfunction]
    fn get_threshold(vm: &VirtualMachine) -> PyObjectRef {
        let (t0, t1, t2) = gc_state::gc_state().get_threshold();
        vm.ctx
            .new_tuple(vec![
                vm.ctx.new_int(t0).into(),
                vm.ctx.new_int(t1).into(),
                vm.ctx.new_int(t2).into(),
            ])
            .into()
    }

    /// Set the collection thresholds.
    #[pyfunction]
    fn set_threshold(threshold0: u32, threshold1: OptionalArg<u32>, threshold2: OptionalArg<u32>) {
        gc_state::gc_state().set_threshold(
            threshold0,
            threshold1.into_option(),
            threshold2.into_option(),
        );
    }

    /// Return the current collection counts as a tuple.
    #[pyfunction]
    fn get_count(vm: &VirtualMachine) -> PyObjectRef {
        let (c0, c1, c2) = gc_state::gc_state().get_count();
        vm.ctx
            .new_tuple(vec![
                vm.ctx.new_int(c0).into(),
                vm.ctx.new_int(c1).into(),
                vm.ctx.new_int(c2).into(),
            ])
            .into()
    }

    /// Return the current debugging flags.
    #[pyfunction]
    fn get_debug() -> u32 {
        gc_state::gc_state().get_debug()
    }

    /// Set the debugging flags.
    #[pyfunction]
    fn set_debug(flags: u32) {
        gc_state::gc_state().set_debug(flags);
    }

    /// Return a list of per-generation gc stats.
    #[pyfunction]
    fn get_stats(vm: &VirtualMachine) -> PyResult<PyListRef> {
        let stats = gc_state::gc_state().get_stats();
        let mut result = Vec::with_capacity(3);

        for stat in stats.iter() {
            let dict = vm.ctx.new_dict();
            dict.set_item("collections", vm.ctx.new_int(stat.collections).into(), vm)?;
            dict.set_item("collected", vm.ctx.new_int(stat.collected).into(), vm)?;
            dict.set_item(
                "uncollectable",
                vm.ctx.new_int(stat.uncollectable).into(),
                vm,
            )?;
            result.push(dict.into());
        }

        Ok(vm.ctx.new_list(result))
    }

    /// Return the list of objects tracked by the collector.
    #[derive(FromArgs)]
    struct GetObjectsArgs {
        #[pyarg(any, optional)]
        generation: OptionalArg<Option<i32>>,
    }

    #[pyfunction]
    fn get_objects(args: GetObjectsArgs, vm: &VirtualMachine) -> PyResult<PyListRef> {
        let generation_opt = args.generation.flatten();
        if let Some(g) = generation_opt {
            if g < 0 || g > 2 {
                return Err(
                    vm.new_value_error(format!("generation must be in range(0, 3), not {}", g))
                );
            }
        }
        let objects = gc_state::gc_state().get_objects(generation_opt);
        Ok(vm.ctx.new_list(objects))
    }

    /// Return the list of objects directly referred to by any of the arguments.
    #[pyfunction]
    fn get_referents(args: FuncArgs, vm: &VirtualMachine) -> PyListRef {
        let mut result = Vec::new();

        for obj in args.args {
            // Use the gc_get_referents method to get references
            result.extend(obj.gc_get_referents());
        }

        vm.ctx.new_list(result)
    }

    /// Return the list of objects that directly refer to any of the arguments.
    #[pyfunction]
    fn get_referrers(args: FuncArgs, vm: &VirtualMachine) -> PyListRef {
        // This is expensive: we need to scan all tracked objects
        // For now, return an empty list (would need full object tracking to implement)
        let _ = args;
        vm.ctx.new_list(vec![])
    }

    /// Return True if the object is tracked by the garbage collector.
    #[pyfunction]
    fn is_tracked(obj: PyObjectRef) -> bool {
        // An object is tracked if it has IS_TRACE = true (has a trace function)
        obj.is_gc_tracked()
    }

    /// Return True if the object has been finalized by the garbage collector.
    #[pyfunction]
    fn is_finalized(_obj: PyObjectRef) -> bool {
        // For now, always return false
        // Full implementation would check the FINALIZED flag in gc_prev
        false
    }

    /// Freeze all objects tracked by gc.
    #[pyfunction]
    fn freeze() {
        gc_state::gc_state().freeze();
    }

    /// Unfreeze all objects in the permanent generation.
    #[pyfunction]
    fn unfreeze() {
        gc_state::gc_state().unfreeze();
    }

    /// Return the number of objects in the permanent generation.
    #[pyfunction]
    fn get_freeze_count() -> usize {
        gc_state::gc_state().get_freeze_count()
    }

    /// gc.garbage - list of uncollectable objects
    #[pyattr]
    fn garbage(vm: &VirtualMachine) -> PyListRef {
        let items = gc_state::gc_state().garbage.lock().clone();
        vm.ctx.new_list(items)
    }

    /// gc.callbacks - list of callbacks to be invoked
    #[pyattr]
    fn callbacks(vm: &VirtualMachine) -> PyListRef {
        let items = gc_state::gc_state().callbacks.lock().clone();
        vm.ctx.new_list(items)
    }

    /// Helper function to invoke GC callbacks
    fn invoke_callbacks(vm: &VirtualMachine, phase: &str, generation: usize) {
        let callbacks = gc_state::gc_state().callbacks.lock().clone();
        if callbacks.is_empty() {
            return;
        }

        let phase_str: PyObjectRef = vm.ctx.new_str(phase).into();
        let info = vm.ctx.new_dict();
        let _ = info.set_item("generation", vm.ctx.new_int(generation).into(), vm);
        let _ = info.set_item("collected", vm.ctx.new_int(0).into(), vm);
        let _ = info.set_item("uncollectable", vm.ctx.new_int(0).into(), vm);

        for callback in callbacks {
            let _ = callback.call((phase_str.clone(), info.clone()), vm);
        }
    }
}
