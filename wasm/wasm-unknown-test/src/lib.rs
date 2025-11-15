use rustpython_vm::{eval, Interpreter};

#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn eval(s: *const u8, l: usize) -> u32 {
    let src = std::slice::from_raw_parts(s, l);
    let src = std::str::from_utf8(src).unwrap();
    Interpreter::without_stdlib(Default::default()).enter(|vm| {
        let res = eval::eval(vm, src, vm.new_scope_with_builtins(), "<string>").unwrap();
        res.try_into_value(vm).unwrap()
    })
}

#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn process() -> i32 {
    let mut settings = rustpython_vm::Settings::default();
    settings.path_list = vec!["Lib".into()];

    println!("10");
    eprintln!("20");

    let interpreter = match Interpreter::debug_init(settings, |vm| {}) {
        Ok(interpreter) => interpreter,
        Err(code) => return code,
    };
    return 30;
    interpreter.enter(|vm| {
        40
        //     // let res = eval::eval(vm, src, vm.new_scope_with_builtins(), "<string>").unwrap();
        //     // res.try_into_value(vm).unwrap()
    })
}
