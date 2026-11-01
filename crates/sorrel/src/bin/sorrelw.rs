//! `sorrelw`: `sorrel` without a console window, for shortcuts and Explorer.
//! Starts the sibling `sorrel` with the same arguments (bare = the launcher)
//! and exits. On Unix it just execs it.
#![cfg_attr(windows, windows_subsystem = "windows")]

use std::process::Command;

fn launch() -> std::io::Result<()> {
    let sorrel =
        std::env::current_exe()?.with_file_name(format!("sorrel{}", std::env::consts::EXE_SUFFIX));
    let mut cmd = Command::new(&sorrel);
    cmd.args(std::env::args_os().skip(1));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(cmd.exec()) // only returns on failure
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use std::process::Stdio;
        // CREATE_NO_WINDOW: a hidden console its own children (the auto
        // sidecar) inherit, instead of a new visible one each. With no
        // terminal on stderr, `sorrel` runs the GUI in-process.
        cmd.creation_flags(0x0800_0000)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(|_| ())
    }
    .map_err(|e| std::io::Error::new(e.kind(), format!("start {}: {e}", sorrel.display())))
}

fn main() {
    if let Err(e) = launch() {
        // No console to print to: leave the reason where `sorrel status` looks.
        if let Ok(dir) = sorrel_data::user_dir() {
            let _ = std::fs::write(dir.join("sorrelw.log"), format!("{e}\n"));
        }
        std::process::exit(1);
    }
}
