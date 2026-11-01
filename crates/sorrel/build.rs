//! Windows resources for both executables: icon, version info and manifest.
//! Nothing happens on other targets, so Linux and macOS builds need no windres.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    let dir = std::path::PathBuf::from(env("CARGO_MANIFEST_DIR")).join("assets");
    let version = env("CARGO_PKG_VERSION");
    // 0.3.1-rc1 -> 0,3,1,0: the numeric prefix of each dotted part.
    let mut nums = version
        .split('.')
        .map(|p| {
            p.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .map(|p| p.parse::<u16>().unwrap_or(0))
        .chain(std::iter::repeat(0));
    let numeric = format!(
        "{},{},{},{}",
        nums.next().unwrap(),
        nums.next().unwrap(),
        nums.next().unwrap(),
        0
    );
    let authors = env("CARGO_PKG_AUTHORS").replace(':', ", ");
    let rc_str = |s: &str| s.replace('\\', "\\\\").replace('"', "\"\"");
    let path = |f: &str| rc_str(&dir.join(f).display().to_string().replace('\\', "/"));
    let rc = format!(
        r#"#pragma code_page(65001)
1 ICON "{icon}"
1 24 "{manifest}"
1 VERSIONINFO
FILEVERSION {numeric}
PRODUCTVERSION {numeric}
FILEOS 0x40004
FILETYPE 1
{{
  BLOCK "StringFileInfo"
  {{
    BLOCK "040904B0"
    {{
      VALUE "CompanyName", "{authors}"
      VALUE "FileDescription", "Sorrel spike-sorting curation"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "sorrel"
      VALUE "LegalCopyright", "Copyright (c) {authors}. Licensed {license}."
      VALUE "ProductName", "Sorrel"
      VALUE "ProductVersion", "{version}"
    }}
  }}
  BLOCK "VarFileInfo"
  {{
    VALUE "Translation", 0x409, 1200
  }}
}}
"#,
        icon = path("sorrel.ico"),
        manifest = path("sorrel.manifest"),
        authors = rc_str(&authors),
        license = rc_str(&env("CARGO_PKG_LICENSE")),
        version = rc_str(&version),
    );
    let out = std::path::PathBuf::from(env("OUT_DIR")).join("sorrel.rc");
    std::fs::write(&out, rc).expect("write sorrel.rc");
    embed_resource::compile_for(&out, ["sorrel", "sorrelw"], embed_resource::NONE)
        .manifest_required()
        .expect("embed Windows resources");
}
