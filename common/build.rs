use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    gen_unicode_aliases(&out_dir);
    Ok(())
}

// Generate unicode aliases names,
// generated from https://www.unicode.org/Public/14.0.0/ucd/NameAliases.txt
fn gen_unicode_aliases(out_dir: &Path) {
    const ALIASES_DATA: &str = include_str!("NameAliases.txt");

    let mut aliases = phf_codegen::Map::new();
    for line in ALIASES_DATA.split('\n') {
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, ';');
        let code = parts.next().unwrap();
        let alias = parts.next().unwrap();
        let _type = parts.next().unwrap();
        let formatted = format!("'\\u{{{code}}}'");
        aliases.entry(alias, &formatted);
    }

    let aliases = aliases.build();
    writeln!(
        BufWriter::new(File::create(out_dir.join("aliases.rs")).unwrap()),
        "{aliases}",
    )
    .unwrap();
}
