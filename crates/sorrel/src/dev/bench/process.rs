//! Bounded subprocesses and isolated model backends. Prompts never appear in
//! command arguments (notably cmd.exe shims); only private prompt-file paths do.

use super::write_json;
use anyhow::{bail, ensure, Context, Result};
use clap::ValueEnum;
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum Backend {
    Opencode,
    Claude,
}

impl Backend {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Opencode => "opencode",
            Self::Claude => "claude",
        }
    }
}

fn stop(child: &mut Child) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill.exe")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-TERM", "--", &format!("-{}", child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
}

fn capture(
    command: &mut Command,
    timeout: Duration,
) -> Result<(Option<std::process::ExitStatus>, Vec<u8>, Vec<u8>)> {
    // Files rather than pipes: a noisy child cannot block waiting for its stderr
    // to be consumed while its parent is waiting for exit.
    let mut out = tempfile::tempfile()?;
    let mut err = tempfile::tempfile()?;
    command
        .stdout(out.try_clone()?)
        .stderr(err.try_clone()?)
        .stdin(Stdio::null());
    group(command);
    let mut child = command.spawn().context("start subprocess")?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if start.elapsed() >= timeout {
            stop(&mut child);
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    out.rewind()?;
    err.rewind()?;
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    out.read_to_end(&mut stdout)?;
    err.read_to_end(&mut stderr)?;
    Ok((status, stdout, stderr))
}

fn captured_text(
    status: Option<std::process::ExitStatus>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    strict: bool,
) -> Result<String> {
    let status = status.context("subprocess timed out")?;
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        bail!(
            "subprocess exited {status}: {}",
            stderr
                .chars()
                .rev()
                .take(600)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        );
    }
    if strict {
        String::from_utf8(stdout).context("subprocess stdout is not UTF-8")
    } else {
        Ok(String::from_utf8_lossy(&stdout).into_owned())
    }
}

pub(super) fn checked(command: &mut Command, timeout: Duration) -> Result<String> {
    let (status, stdout, stderr) = capture(command, timeout)?;
    // Structured stdout (scores, evidence, model events) stays strict.
    captured_text(status, stdout, stderr, true)
}

/// Sidecar progress/log output is not a JSON payload and may use the Windows
/// code page. Preserve exact bytes (including on errors/timeouts) before any
/// decoding; byte encoding alone cannot promote or fail a curation run.
pub(super) fn checked_logged(
    command: &mut Command,
    timeout: Duration,
    directory: &Path,
) -> Result<()> {
    let (status, stdout, stderr) = capture(command, timeout)?;
    fs::write(directory.join("auto.stdout"), &stdout)?;
    fs::write(directory.join("auto.stderr"), &stderr)?;
    captured_text(status, stdout, stderr, false)?;
    Ok(())
}

fn executable(name: &str) -> PathBuf {
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        #[cfg(windows)]
        let suffixes = [".exe", ".cmd", ".bat"];
        #[cfg(not(windows))]
        let suffixes = [""];
        for suffix in suffixes {
            let p = dir.join(format!("{name}{suffix}"));
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(name)
}

pub(super) fn opencode_config(sorrel: &Path, data: &Path) -> Value {
    json!({
        "$schema": "https://opencode.ai/config.json",
        "snapshots": false,
        "default_agent": "curation-bench",
        "agents": {"curation-bench": {
            "mode": "primary",
            "system": "You curate spike sorting through the sorrel MCP server. Follow the supplied review protocol. Only sorrel tools are available. Never seek human labels, source folders, or other sessions.",
            "steps": 120,
            "permissions": [
                {"action":"*","resource":"*","effect":"deny"},
                {"action":"sorrel_*","resource":"*","effect":"allow"}
            ]
        }},
        "mcp": {"servers": {"sorrel": {
            "type":"local", "command":[sorrel, "mcp", data], "codemode": false
        }}}
    })
}

pub(super) fn auto_config() -> Value {
    json!({"$schema":"https://opencode.ai/config.json", "snapshots":false,
    "default_agent":"curation-auto", "agents":{"curation-auto":{
        "mode":"primary", "system":"Classify only the supplied spike-sorting evidence. Do not use tools.",
        "permissions":[{"action":"*","resource":"*","effect":"deny"}]
    }}})
}

fn isolated_directory(command: &mut Command, directory: &Path) {
    // OpenCode respects PWD when choosing its Location. current_dir alone
    // leaves the harness's repository PWD inherited on Windows.
    command
        .current_dir(directory)
        .env("PWD", directory)
        .env_remove("OPENCODE_SESSION_ID");
}

#[derive(Default)]
pub(super) struct Review {
    pub cost: f64,
    pub turns: u64,
    pub tool_calls: u64,
    pub model: Option<String>,
    pub usage: Value,
    pub error: Option<String>,
}

impl Review {
    fn event(&mut self, event: &Value) {
        match event["type"].as_str().unwrap_or("") {
            "step_finish" => {
                self.turns += 1;
                self.cost += event["part"]["cost"].as_f64().unwrap_or(0.0);
                add_usage(&mut self.usage, &event["part"]["tokens"]);
            }
            "tool_use" => {
                if event["part"]["tool"]
                    .as_str()
                    .is_some_and(|t| t.starts_with("sorrel_"))
                {
                    self.tool_calls += 1;
                }
            }
            "error" => {
                self.error = Some(event["error"].to_string());
            }
            _ => {}
        }
        if let Some(m) = event["model"].as_str() {
            self.model = Some(m.to_owned());
        }
    }
}

fn add_usage(total: &mut Value, next: &Value) {
    if let Some(count) = next.as_u64() {
        *total = json!(total.as_u64().unwrap_or(0).saturating_add(count));
    } else if let Some(fields) = next.as_object() {
        if !total.is_object() {
            *total = json!({});
        }
        for (key, value) in fields {
            add_usage(&mut total[key], value);
        }
    }
}

pub(super) fn review(
    backend: Backend,
    root: &Path,
    sorrel: &Path,
    data: &Path,
    prompt: &str,
    model: Option<&str>,
    cap: f64,
    timeout: Duration,
    env: &[(String, Option<String>)],
) -> Result<Review> {
    ensure!(
        cap.is_finite() && cap > 0.0,
        "per-run cap must be finite and positive"
    );
    let cwd = tempfile::Builder::new()
        .prefix("sorrel-bench-agent-")
        .tempdir()?;
    let prompt_path = cwd.path().join("review.md");
    fs::write(&prompt_path, prompt)?;
    let mut c = Command::new(executable(backend.name()));
    isolated_directory(&mut c, cwd.path());
    for (key, value) in env {
        if let Some(value) = value {
            c.env(key, value);
        } else {
            c.env_remove(key);
        }
    }
    match backend {
        Backend::Opencode => {
            let config = opencode_config(sorrel, data);
            write_json(&cwd.path().join("opencode.json"), &config)?;
            write_json(&root.join("agent-config.json"), &config)?;
            let mut check = Command::new(executable("opencode"));
            isolated_directory(&mut check, cwd.path());
            check.args(["--standalone", "debug", "agents"]);
            let agents: Value =
                serde_json::from_str(&checked(&mut check, timeout.min(Duration::from_secs(120)))?)?;
            ensure!(
                agents
                    .as_array()
                    .is_some_and(|rows| rows.iter().any(|a| a["id"] == "curation-bench")),
                "OpenCode did not load the isolated review agent; no model call was made"
            );
            c.args([
                "--standalone",
                "run",
                "--format",
                "json",
                "--agent",
                "curation-bench",
            ]);
            if let Some(m) = model {
                c.args(["--model", m]);
            }
            c.arg("Follow the attached review protocol, apply your curation decisions with sorrel tools, then stop.").arg("--file").arg(&prompt_path);
        }
        Backend::Claude => {
            let mcp = cwd.path().join("mcp.json");
            write_json(
                &mcp,
                &json!({"mcpServers":{"sorrel":{"command":sorrel,"args":["mcp",data]}}}),
            )?;
            c.args([
                "-p",
                "--output-format",
                "json",
                "--strict-mcp-config",
                "--mcp-config",
            ])
            .arg(mcp)
            .args([
                "--setting-sources",
                "project",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--tools",
                "",
                "--allowedTools",
                "mcp__sorrel__*",
                "--max-budget-usd",
            ])
            .arg(cap.to_string())
            .arg("--system-prompt-file")
            .arg(&prompt_path);
            if let Some(m) = model {
                c.args(["--model", m]);
            }
            c.arg(
                "Review the dataset behind the sorrel MCP server, apply your decisions, then stop.",
            );
        }
    }
    let stderr = fs::File::create(root.join("agent.stderr"))?;
    let mut log = fs::File::create(root.join(if matches!(backend, Backend::Claude) {
        "claude.json"
    } else {
        "opencode.jsonl"
    }))?;
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(stderr);
    group(&mut c);
    let mut child = c.spawn().context("start curation backend")?;
    let stdout = child.stdout.take().context("backend has no stdout")?;
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let start = Instant::now();
    let mut result = Review {
        model: model.map(str::to_owned),
        ..Default::default()
    };
    let mut raw = String::new();
    loop {
        if start.elapsed() >= timeout {
            result.error = Some("timeout".into());
            stop(&mut child);
            break;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                let line = line?;
                writeln!(log, "{line}")?;
                if matches!(backend, Backend::Claude) {
                    raw.push_str(&line);
                    raw.push('\n');
                } else if let Ok(event) = serde_json::from_str::<Value>(&line) {
                    result.event(&event);
                }
                if result.cost >= cap {
                    result.error = Some("budget cap reached".into());
                    stop(&mut child);
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() && result.error.is_none() {
                result.error = Some(format!("backend exited {status}; inspect agent.stderr"));
            }
            break;
        }
        if start.elapsed() >= timeout {
            result.error.get_or_insert_with(|| "timeout".into());
            stop(&mut child);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(rx);
    // stop() closes the complete process tree on Windows, including MCP hosts.
    let _ = reader.join();
    if matches!(backend, Backend::Claude) {
        let v: Value = serde_json::from_str(&raw).context("invalid Claude JSON")?;
        result.cost = v["total_cost_usd"].as_f64().unwrap_or(0.0);
        result.turns = v["num_turns"].as_u64().unwrap_or(0);
        result.usage = v["usage"].clone();
        if v["is_error"] == true {
            result.error = Some(v["result"].to_string());
        }
        // Tool connectivity is checked from the saved curation journal by caller.
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "subprocess fixture, invoked only by raw_log_capture test"]
    fn raw_log_fixture() {
        let Ok(code) = std::env::var("SORREL_TEST_RAW_LOG_EXIT") else {
            return;
        };
        std::io::stdout().write_all(b"progress\xff\n").unwrap();
        std::io::stderr()
            .write_all(b"underlying failure\xfe\n")
            .unwrap();
        std::io::stdout().flush().unwrap();
        std::io::stderr().flush().unwrap();
        std::process::exit(code.parse().unwrap());
    }

    fn raw_log_child(code: &str) -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args([
            "--exact",
            "dev::bench::process::tests::raw_log_fixture",
            "--ignored",
            "--nocapture",
        ])
        .env("SORREL_TEST_RAW_LOG_EXIT", code);
        c
    }

    #[test]
    fn raw_log_capture_preserves_bytes_and_never_masks_nonzero_exit() {
        let d = tempfile::tempdir().unwrap();
        checked_logged(&mut raw_log_child("0"), Duration::from_secs(10), d.path()).unwrap();
        assert!(fs::read(d.path().join("auto.stdout"))
            .unwrap()
            .contains(&0xff));
        assert!(fs::read(d.path().join("auto.stderr"))
            .unwrap()
            .contains(&0xfe));
        assert!(
            checked(&mut raw_log_child("0"), Duration::from_secs(10)).is_err(),
            "structured stdout must remain strict UTF-8"
        );
        let error = checked_logged(&mut raw_log_child("7"), Duration::from_secs(10), d.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("underlying failure"), "{error}");
        assert!(
            error.contains("exited"),
            "a failure must never become success: {error}"
        );
        assert!(fs::read(d.path().join("auto.stdout"))
            .unwrap()
            .contains(&0xff));
    }

    #[test]
    fn only_sorrel_is_exposed_and_cost_is_accumulated() {
        let cfg = opencode_config(Path::new("sorrel.exe"), Path::new("private/input"));
        assert_eq!(cfg["mcp"]["servers"].as_object().unwrap().len(), 1);
        assert_eq!(cfg["mcp"]["servers"]["sorrel"]["codemode"], false);
        let p = &cfg["agents"]["curation-bench"]["permissions"];
        assert_eq!(p[0]["effect"], "deny");
        assert_eq!(p[1]["action"], "sorrel_*");
        let mut r = Review::default();
        for _ in 0..2 {
            r.event(&json!({"type":"step_finish","part":{"cost":0.25}}));
        }
        r.event(&json!({"type":"tool_use","part":{"tool":"sorrel_brief"}}));
        assert_eq!((r.turns, r.tool_calls, r.cost), (2, 1, 0.5));
    }

    #[test]
    fn isolated_cwd_also_replaces_inherited_pwd() {
        let path = Path::new("private-agent");
        let mut command = Command::new("opencode");
        isolated_directory(&mut command, path);
        assert_eq!(command.get_current_dir(), Some(path));
        assert!(command
            .get_envs()
            .any(|(key, value)| key == "PWD" && value == Some(path.as_os_str())));
        let config = auto_config();
        assert_eq!(
            config["agents"]["curation-auto"]["permissions"][0]["effect"],
            "deny"
        );
    }

    #[test]
    fn usage_sums_every_turn_including_nested_cache_tokens() {
        let mut review = Review::default();
        for _ in 0..3 {
            review.event(&json!({"type":"step_finish","part":{"cost":0.1,
                "tokens":{"input":100,"output":20,"reasoning":5,"cache":{"read":70,"write":0}}}}));
        }
        assert_eq!(review.usage["input"], 300);
        assert_eq!(review.usage["output"], 60);
        assert_eq!(review.usage["reasoning"], 15);
        assert_eq!(review.usage["cache"]["read"], 210);
        assert_eq!(review.turns, 3);
    }
}
