//! Tauri shell: launches the bundled Python backend and relays its line protocol.
//!
//! The backend prints `@@MTC@@{json}` lines on stdout; those become `backend-event`
//! events in the webview. Everything else (stderr, stray stdout, Julia output)
//! becomes `backend-log`. Commands from the UI are written to the backend's stdin
//! as JSON lines.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const SENTINEL: &str = "@@MTC@@";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Serialize, Clone, Debug)]
struct RuntimeInfo {
    runtime: String,
    backend: String,
    workdir: String,
    python: String,
    dev: bool,
}

struct Backend {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Default)]
struct AppState {
    backend: Mutex<Option<Backend>>,
    info: Mutex<Option<RuntimeInfo>>,
    #[cfg(windows)]
    job: Mutex<Option<job::Job>>,
}

// ------------------------------------------------------------------ job object
// All backend processes (Python coordinator + Julia worker) join a job object that
// is killed when the app exits, so nothing is left running after a crash.
#[cfg(windows)]
mod job {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn new() -> Option<Job> {
            unsafe {
                let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if h.is_null() {
                    return None;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    h,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    CloseHandle(h);
                    return None;
                }
                Some(Job(h))
            }
        }

        pub fn assign(&self, child: &std::process::Child) -> bool {
            use std::os::windows::io::AsRawHandle;
            unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as HANDLE) != 0 }
        }
    }
}

// ------------------------------------------------------------------ paths

fn resolve_runtime(app: &AppHandle) -> Result<(PathBuf, PathBuf, bool), String> {
    if let Ok(rt) = std::env::var("MESHTOCAD_RUNTIME") {
        let rt = PathBuf::from(rt);
        let be = std::env::var("MESHTOCAD_BACKEND").map(PathBuf::from).unwrap_or(rt.join("backend"));
        return Ok((rt, be, true));
    }
    #[cfg(debug_assertions)]
    {
        // `tauri dev`: use the repo's runtime and the live backend sources.
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let rt = repo.join("runtime");
        if rt.join("python").join("python.exe").exists() {
            return Ok((rt, repo.join("backend"), true));
        }
    }
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let rt = res.join("runtime");
    Ok((rt.clone(), rt.join("backend"), false))
}

fn clean(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

fn prune_old_sessions(work: &Path, keep: usize) {
    if let Ok(rd) = std::fs::read_dir(work) {
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("session-")))
            .collect();
        dirs.sort();
        if dirs.len() > keep {
            for d in &dirs[..dirs.len() - keep] {
                let _ = std::fs::remove_dir_all(d);
            }
        }
    }
}

/// Read a pipe line by line. Invalid UTF-8 is replaced instead of ending the loop:
/// if we ever stopped reading, the pipe would fill up and block the backend.
fn for_each_line<R: std::io::Read>(r: R, mut f: impl FnMut(&str)) {
    let mut reader = BufReader::new(r);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf);
        f(line.trim_end_matches(|c| c == '\r' || c == '\n'));
    }
}

// ------------------------------------------------------------------ backend

fn spawn_backend(app: &AppHandle, state: &AppState) -> Result<RuntimeInfo, String> {
    let (rt, be, dev) = resolve_runtime(app)?;
    let python = rt.join("python").join("python.exe");
    if !python.exists() {
        return Err(format!("Bundled runtime not found at {}", clean(&rt)));
    }
    let data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    let work_root = data.join("work");
    std::fs::create_dir_all(&work_root).map_err(|e| e.to_string())?;
    prune_old_sessions(&work_root, 4);
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let workdir = work_root.join(format!("session-{stamp}"));
    std::fs::create_dir_all(&workdir).map_err(|e| e.to_string())?;
    let user_depot = data.join("julia_depot");
    std::fs::create_dir_all(&user_depot).map_err(|e| e.to_string())?;
    // Julia's own depots must stay in the list: they hold the shipped stdlib caches.
    let jl = rt.join("julia");
    let depot_path = [
        clean(&user_depot),
        clean(&rt.join("julia_depot")),
        clean(&jl.join("local").join("share").join("julia")),
        clean(&jl.join("share").join("julia")),
    ]
    .join(";");

    let mut cmd = Command::new(&python);
    cmd.args(["-X", "utf8", "-u", "-m", "meshtocad.server"]);
    for k in ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX", "JULIA_PROJECT",
              "JULIA_LOAD_PATH", "JULIA_DEPOT_PATH", "JULIA_BINDIR"] {
        cmd.env_remove(k);
    }
    cmd.env("PYTHONPATH", clean(&be))
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHON_JULIAPKG_PROJECT", clean(&rt.join("julia_env")))
        .env("PYTHON_JULIAPKG_EXE", clean(&rt.join("julia").join("bin").join("julia.exe")))
        .env("PYTHON_JULIAPKG_OFFLINE", "yes")
        .env("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
        .env("PYTHON_JULIACALL_THREADS", "auto")
        .env("JULIA_DEPOT_PATH", depot_path)
        .env("JULIA_PKG_OFFLINE", "true")
        .env("MESHTOCAD_WORKDIR", clean(&workdir))
        .current_dir(&workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd.spawn().map_err(|e| format!("Failed to start backend: {e}"))?;
    #[cfg(windows)]
    {
        let mut job = state.job.lock().unwrap();
        if job.is_none() {
            *job = job::Job::new();
        }
        if let Some(j) = job.as_ref() {
            j.assign(&child);
        }
    }
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    let app_out = app.clone();
    std::thread::spawn(move || {
        for_each_line(stdout, |line| {
            if let Some(json) = line.strip_prefix(SENTINEL) {
                match serde_json::from_str::<serde_json::Value>(json) {
                    Ok(v) => {
                        let _ = app_out.emit("backend-event", v);
                    }
                    Err(e) => {
                        let _ = app_out.emit("backend-log", format!("[protocol error] {e}"));
                    }
                }
            } else if !line.trim().is_empty() {
                let _ = app_out.emit("backend-log", line.to_string());
            }
        });
        let _ = app_out.emit("backend-exit", ());
    });
    let app_err = app.clone();
    std::thread::spawn(move || {
        for_each_line(stderr, |line| {
            if !line.trim().is_empty() {
                let _ = app_err.emit("backend-log", line.to_string());
            }
        });
    });

    let info = RuntimeInfo {
        runtime: clean(&rt),
        backend: clean(&be),
        workdir: clean(&workdir),
        python: clean(&python),
        dev,
    };
    *state.backend.lock().unwrap() = Some(Backend { child, stdin });
    *state.info.lock().unwrap() = Some(info.clone());
    Ok(info)
}

fn stop_backend(state: &AppState) {
    if let Some(mut b) = state.backend.lock().unwrap().take() {
        drop(b.stdin); // EOF -> backend shuts down its worker and exits
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(4) {
            if let Ok(Some(_)) = b.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = b.child.kill();
        let _ = b.child.wait();
    }
}

// ------------------------------------------------------------------ commands

#[tauri::command]
fn backend_start(app: AppHandle, state: State<'_, AppState>) -> Result<RuntimeInfo, String> {
    {
        let mut guard = state.backend.lock().unwrap();
        if let Some(b) = guard.as_mut() {
            if let Ok(None) = b.child.try_wait() {
                if let Some(info) = state.info.lock().unwrap().clone() {
                    return Ok(info);
                }
            }
            *guard = None;
        }
    }
    spawn_backend(&app, &state)
}

#[tauri::command]
fn backend_restart(app: AppHandle, state: State<'_, AppState>) -> Result<RuntimeInfo, String> {
    stop_backend(&state);
    spawn_backend(&app, &state)
}

#[tauri::command]
fn backend_send(state: State<'_, AppState>, msg: serde_json::Value) -> Result<(), String> {
    let mut guard = state.backend.lock().unwrap();
    let b = guard.as_mut().ok_or("Backend is not running")?;
    let line = serde_json::to_string(&msg).map_err(|e| e.to_string())?;
    b.stdin
        .write_all(line.as_bytes())
        .and_then(|_| b.stdin.write_all(b"\n"))
        .and_then(|_| b.stdin.flush())
        .map_err(|e| format!("Backend pipe closed: {e}"))
}

/// Raw bytes of a file the backend wrote into its session folder.
#[tauri::command]
fn read_binary(state: State<'_, AppState>, path: String) -> Result<tauri::ipc::Response, String> {
    let info = state.info.lock().unwrap().clone().ok_or("Backend not started")?;
    let work = std::fs::canonicalize(&info.workdir).map_err(|e| e.to_string())?;
    let p = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
    if !p.starts_with(&work) {
        return Err("Refusing to read outside the session folder".into());
    }
    let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// Open Explorer with the file selected (or the folder opened).
#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    #[cfg(windows)]
    {
        let mut c = Command::new("explorer.exe");
        if p.is_file() {
            c.arg(format!("/select,{}", path));
        } else {
            c.arg(&path);
        }
        c.spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(not(windows))]
    {
        let _ = p;
    }
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            backend_start,
            backend_restart,
            backend_send,
            read_binary,
            reveal_path
        ])
        .build(tauri::generate_context!())
        .expect("error while building PyCadSR");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            stop_backend(&handle.state::<AppState>());
        }
    });
}
