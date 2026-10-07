from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dots_tts_lab.fuxuan_batch import (
    BatchCancelled,
    discover_text_files,
    load_runtime,
    run_batch,
)
from dots_tts_lab.postprocess import load_edge_trim_config, load_voice_polish_config
from dots_tts_lab.voice_registry import (
    DEFAULT_VOICE_REGISTRY_PATH,
    LoadedVoiceRegistry,
    VoiceProfile,
    load_voice_registry,
    validate_voice_profile_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
WEBUI_DIR = ROOT / "apps/fuxuan_webui"
INDEX_HTML = WEBUI_DIR / "index.html"
STATE_DIR = ROOT / "outputs/fuxuan_webui"
STATE_FILE = STATE_DIR / "state.json"
HISTORY_FILE = STATE_DIR / "history.jsonl"
SERVER_FILE = STATE_DIR / "server.json"
WORK_DIR = STATE_DIR / "work"

APP_ID = "dots-tts-local-voice-webui-v1"
SCHEMA_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7868
PORT_ATTEMPTS = 10
CLOSE_GRACE_SECONDS = 3.0

DEFAULT_MODEL_ID = load_voice_registry(
    DEFAULT_VOICE_REGISTRY_PATH,
    validate_artifacts=False,
).default_model_id

ACTIVE_JOB_STATES = frozenset(
    {"preparing", "loading_model", "generating", "finalizing"}
)
RECOVERABLE_STATES = ACTIVE_JOB_STATES | {"starting"}
ALLOWED_TRANSITIONS = {
    "starting": {"idle", "failed", "shutting_down"},
    "idle": {"preparing", "failed", "shutting_down"},
    "preparing": {"loading_model", "failed", "interrupted"},
    "loading_model": {"generating", "failed", "interrupted"},
    "generating": {"generating", "finalizing", "failed", "interrupted"},
    "finalizing": {"generating", "completed", "failed", "interrupted"},
    "completed": {"preparing", "shutting_down"},
    "failed": {"preparing", "shutting_down"},
    "interrupted": {"shutting_down"},
    "shutting_down": set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class StateStore:
    def __init__(self, state_file: Path, history_file: Path) -> None:
        self.state_file = state_file
        self.history_file = history_file
        self._lock = threading.RLock()
        self._state: dict[str, Any] = self._read_disk_state()

    def _read_disk_state(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _persist(self, state: dict[str, Any], *, append_history: bool) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(self.state_file)
        if append_history:
            with self.history_file.open("a", encoding="utf-8") as history:
                history.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")

    def replace(self, state: dict[str, Any], *, append_history: bool = True) -> None:
        with self._lock:
            state = dict(state)
            state["updated_at"] = _utc_now()
            self._state = state
            self._persist(self._state, append_history=append_history)

    def transition(self, status: str, message: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            current = str(self._state.get("status", "starting"))
            if status != current and status not in ALLOWED_TRANSITIONS.get(current, set()):
                raise RuntimeError(f"Invalid WebUI state transition: {current} -> {status}")
            next_state = dict(self._state)
            next_state.update(updates)
            next_state.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "message": message,
                    "updated_at": _utc_now(),
                }
            )
            self._state = next_state
            self._persist(self._state, append_history=True)
            return dict(self._state)


def cleanup_stale_artifacts(
    state_dir: Path,
    output_dir: Path | list[Path] | tuple[Path, ...],
) -> list[str]:
    removed: list[str] = []
    work_dir = state_dir / "work"
    if work_dir.exists() and _is_within(work_dir, state_dir):
        removed.extend(str(path) for path in work_dir.rglob("*") if path.is_file())
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    output_dirs = [output_dir] if isinstance(output_dir, Path) else list(output_dir)
    for directory in output_dirs:
        if directory.is_dir():
            for partial in directory.rglob("*.partial.wav"):
                if partial.is_file() and _is_within(partial, directory):
                    partial.unlink()
                    removed.append(str(partial))

    temporary_state = state_dir / "state.tmp"
    if temporary_state.is_file() and _is_within(temporary_state, state_dir):
        temporary_state.unlink()
        removed.append(str(temporary_state))
    return removed


def recover_and_start_session(
    store: StateStore,
    *,
    state_dir: Path,
    output_dir: Path | list[Path] | tuple[Path, ...],
    default_model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    previous = store.snapshot()
    previous_status = str(previous.get("status", "unknown"))
    recovered_job_id = previous.get("job_id")
    if previous_status in RECOVERABLE_STATES:
        interrupted = dict(previous)
        interrupted.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "interrupted",
                "message": "检测到上次运行未正常结束，已在本次启动时恢复。",
                "interrupted_at": _utc_now(),
            }
        )
        store.replace(interrupted)

    removed = cleanup_stale_artifacts(state_dir, output_dir)
    session_id = uuid.uuid4().hex
    store.replace(
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "job_id": None,
            "status": "starting",
            "message": "正在启动 WebUI…",
            "progress": 0.0,
            "selected_model": default_model_id,
            "outputs": [],
            "errors": [],
            "started_at": _utc_now(),
            "recovery": {
                "previous_status": previous_status,
                "previous_job_id": recovered_job_id,
                "cleaned_artifact_count": len(removed),
                "cleaned_artifacts": removed,
            },
        }
    )
    return store.transition(
        "idle",
        (
            f"准备就绪；已清理 {len(removed)} 个上次遗留的中间产物。"
            if removed
            else "准备就绪。"
        ),
        progress=0.0,
    )


class FuxuanWebApplication:
    def __init__(
        self,
        store: StateStore,
        *,
        registry: LoadedVoiceRegistry | None = None,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.registry = registry or load_voice_registry()
        self._input_dir_override = input_dir
        self._output_dir_override = output_dir
        self._runtime: Any | None = None
        self._runtime_model_id: str | None = None
        self._worker: threading.Thread | None = None
        self._job_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.server: ThreadingHTTPServer | None = None
        self.started_monotonic = time.monotonic()
        self.last_heartbeat: float | None = None
        self.close_requested_at: float | None = None
        self.ui_connected = False

    def public_status(self) -> dict[str, Any]:
        state = self.store.snapshot()
        selected = str(state.get("selected_model") or self.registry.default_model_id)
        if selected not in self.registry.profiles:
            selected = self.registry.default_model_id
        profile = self.registry.get(selected)
        input_dir, output_dir = self._profile_io(profile)
        state["models"] = self.registry.public_models()
        state["default_model"] = self.registry.default_model_id
        state["input_dir"] = str(input_dir)
        state["output_dir"] = str(output_dir)
        state["can_open_output"] = bool(state.get("outputs")) and state.get(
            "status"
        ) == "completed"
        return state

    def mark_page_loaded(self) -> None:
        self.ui_connected = True
        self.last_heartbeat = time.monotonic()

    def heartbeat(self) -> None:
        self.ui_connected = True
        self.last_heartbeat = time.monotonic()
        self.close_requested_at = None

    def request_page_close(self) -> None:
        self.close_requested_at = time.monotonic()

    def start_generation(self, model_id: str) -> tuple[bool, str]:
        if model_id not in self.registry.profiles:
            return False, "未知模型。"
        with self._job_lock:
            if self._worker is not None and self._worker.is_alive():
                return False, "已有 TTS 任务正在运行。"
            if self.shutdown_event.is_set():
                return False, "WebUI 正在关闭。"
            self.cancel_event.clear()
            job_id = uuid.uuid4().hex
            profile = self.registry.get(model_id)
            input_dir, output_dir = self._profile_io(profile)
            self.store.transition(
                "preparing",
                "正在检查输入文件…",
                job_id=job_id,
                selected_model=model_id,
                selected_profile_version=profile.profile_version,
                selected_profile_sha256=profile.canonical_sha256(),
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                progress=1.0,
                outputs=[],
                errors=[],
                job_started_at=_utc_now(),
            )
            self._worker = threading.Thread(
                target=self._run_generation,
                args=(model_id,),
                name="voice-tts-worker",
                daemon=True,
            )
            self._worker.start()
        return True, "任务已开始。"

    def _profile_io(self, profile: VoiceProfile) -> tuple[Path, Path]:
        input_dir = self._input_dir_override or self.registry.path(profile.io.input_dir)
        output_dir = self._output_dir_override or self.registry.path(profile.io.output_dir)
        return input_dir.resolve(), output_dir.resolve()

    def _release_runtime(self) -> None:
        self._runtime = None
        self._runtime_model_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_selected_runtime(self, model_id: str) -> Any:
        profile = self.registry.get(model_id)
        if self._runtime is not None and self._runtime_model_id in {None, model_id}:
            self._runtime_model_id = model_id
            return self._runtime
        if self._runtime is not None:
            self._release_runtime()
        validate_voice_profile_artifacts(self.registry, profile)
        runtime = profile.runtime
        self._runtime = load_runtime(
            base_model=self.registry.path(profile.base_model.path),
            base_revision=profile.base_model.revision,
            adapter=(
                self.registry.path(profile.adapter.path)
                if profile.adapter is not None
                else None
            ),
            precision=runtime.precision,
            optimize=runtime.optimize,
            max_generate_length=runtime.max_generate_length,
            max_sequence_length=runtime.max_sequence_length,
            vocoder_merge_steps=runtime.vocoder_merge_steps,
            warmup_on_optimize=runtime.warmup_on_optimize,
            merge_lora=runtime.merge_lora,
        )
        self._runtime_model_id = model_id
        return self._runtime

    def _progress(self, event: dict[str, Any]) -> None:
        if self.cancel_event.is_set():
            return
        stage = str(event.get("stage", ""))
        file_index = int(event.get("file_index", 1))
        file_count = max(1, int(event.get("file_count", 1)))
        segment_index = int(event.get("segment_index", 0))
        segment_count = max(1, int(event.get("segment_count", 1)))
        input_name = str(event.get("input", ""))

        if stage in {"generating_segment", "segment_completed"}:
            segment_fraction = segment_index / segment_count
            progress = 8.0 + 87.0 * (
                (file_index - 1 + segment_fraction) / file_count
            )
            self.store.transition(
                "generating",
                f"正在生成 {input_name}：第 {segment_index}/{segment_count} 句",
                progress=min(progress, 95.0),
                current_file=input_name,
                file_index=file_index,
                file_count=file_count,
                segment_index=segment_index,
                segment_count=segment_count,
            )
        elif stage == "finalizing_file":
            progress = 8.0 + 87.0 * (file_index / file_count)
            self.store.transition(
                "finalizing",
                f"正在合并并保存 {input_name}…",
                progress=min(progress, 96.0),
                current_file=input_name,
                file_index=file_index,
                file_count=file_count,
            )

    def _run_generation(self, model_id: str) -> None:
        try:
            profile = self.registry.get(model_id)
            input_dir, output_dir = self._profile_io(profile)
            files = discover_text_files(input_dir)
            if not files:
                self.store.transition(
                    "failed",
                    f"未找到 TXT。请先放入：{input_dir}",
                    progress=0.0,
                    errors=["input_directory_empty"],
                    job_finished_at=_utc_now(),
                )
                return
            if self.cancel_event.is_set():
                raise BatchCancelled("Cancelled before model loading")

            cached = self._runtime is not None and self._runtime_model_id in {
                None,
                model_id,
            }
            self.store.transition(
                "loading_model",
                "正在复用已加载模型…" if cached else "首次运行，正在加载模型…",
                progress=5.0,
                file_count=len(files),
            )
            runtime = self._load_selected_runtime(model_id)
            if self.cancel_event.is_set():
                raise BatchCancelled("Cancelled after model loading")

            self.store.transition(
                "generating",
                f"模型已就绪，开始处理 {len(files)} 个 TXT。",
                progress=8.0,
            )
            summary = run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=load_edge_trim_config(
                    self.registry.path(profile.postprocess.edge_trim_config)
                ),
                base_seed=profile.generation.base_seed,
                pause_ms=profile.generation.pause_ms,
                max_chars=profile.generation.max_chars,
                num_steps=profile.generation.num_steps,
                prompt_audio_path=self.registry.path(profile.prompt.audio_path),
                prompt_text=profile.prompt.text,
                language=profile.generation.language,
                template_name=profile.generation.template_name,
                normalize_text=profile.generation.normalize_text,
                soften_emphasis=profile.generation.soften_emphasis,
                speaker_scale=profile.generation.speaker_scale,
                ode_method=profile.generation.ode_method,
                guidance_scale=profile.generation.guidance_scale,
                force=True,
                progress_callback=self._progress,
                cancelled=self.cancel_event.is_set,
                voice_polish_config=load_voice_polish_config(
                    self.registry.path(profile.postprocess.voice_polish_config)
                ),
            )
            if self.cancel_event.is_set():
                raise BatchCancelled("Cancelled after batch generation")
            outputs = [item["output"] for item in summary["generated"]]
            if summary["errors"]:
                self.store.transition(
                    "failed",
                    f"生成结束，但有 {len(summary['errors'])} 个文件失败。",
                    progress=100.0,
                    outputs=outputs,
                    errors=summary["errors"],
                    job_finished_at=_utc_now(),
                )
                return
            self.store.transition(
                "completed",
                f"完成：已生成 {len(outputs)} 个最终音频。",
                progress=100.0,
                outputs=outputs,
                errors=[],
                job_finished_at=_utc_now(),
            )
        except BatchCancelled:
            current = self.store.snapshot().get("status")
            if current in ACTIVE_JOB_STATES:
                self.store.transition(
                    "interrupted",
                    "任务已中断；中间产物将在下次启动时再次清理。",
                    errors=["cancelled"],
                    job_finished_at=_utc_now(),
                )
        except Exception as error:
            current = self.store.snapshot().get("status")
            if current in ACTIVE_JOB_STATES:
                self.store.transition(
                    "failed",
                    f"生成失败：{type(error).__name__}: {error}",
                    errors=[f"{type(error).__name__}: {error}"],
                    job_finished_at=_utc_now(),
                )

    def open_output_directory(self) -> tuple[bool, str]:
        selected = str(
            self.store.snapshot().get("selected_model")
            or self.registry.default_model_id
        )
        profile = self.registry.get(selected)
        _, output_dir = self._profile_io(profile)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(output_dir))  # type: ignore[attr-defined]
        except (AttributeError, OSError) as error:
            return False, f"无法打开输出目录：{error}"
        return True, "已打开输出目录。"

    def request_shutdown(self, reason: str) -> None:
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        self.cancel_event.set()
        current = str(self.store.snapshot().get("status", "idle"))
        try:
            if current in ACTIVE_JOB_STATES:
                self.store.transition(
                    "interrupted",
                    f"WebUI 已关闭，任务被中断（{reason}）。",
                    errors=["webui_closed"],
                    job_finished_at=_utc_now(),
                )
            elif current != "shutting_down":
                self.store.transition(
                    "shutting_down",
                    "WebUI 已关闭，服务正在退出。",
                )
        finally:
            output_dirs = self.registry.output_dirs()
            if self._output_dir_override is not None:
                output_dirs.append(self._output_dir_override.resolve())
            cleanup_stale_artifacts(self.store.state_file.parent, output_dirs)
            self._release_runtime()
            if self.server is not None:
                self.server.shutdown()

    def watchdog(self) -> None:
        # 生命周期由启动它的终端窗口管理（关窗即杀进程），心跳不再触发关闭；
        # 仅保留页面“关闭服务器”按钮的主动关停。
        while not self.shutdown_event.wait(1.0):
            now = time.monotonic()
            if (
                self.close_requested_at is not None
                and now - self.close_requested_at >= CLOSE_GRACE_SECONDS
            ):
                self.request_shutdown("页面关闭")
                return


class WebUIHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app: FuxuanWebApplication,
    ) -> None:
        self.app = app
        super().__init__(server_address, WebUIRequestHandler)


class WebUIRequestHandler(BaseHTTPRequestHandler):
    server: WebUIHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = min(int(self.headers.get("Content-Length", "0")), 16_384)
        if not length:
            return {}
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            try:
                body = INDEX_HTML.read_bytes()
            except OSError as error:
                self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.server.app.mark_page_loaded()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json(self.server.app.public_status())
        elif path == "/api/health":
            self._send_json({"app_id": APP_ID, "status": "ok"})
        else:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/heartbeat":
            self.server.app.heartbeat()
            self._send_json({"ok": True})
        elif path == "/api/close":
            self.server.app.request_page_close()
            self._send_json({"ok": True})
        elif path == "/api/generate":
            payload = self._read_json()
            accepted, message = self.server.app.start_generation(
                str(payload.get("model", DEFAULT_MODEL_ID))
            )
            status = HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT
            self._send_json({"ok": accepted, "message": message}, status)
        elif path == "/api/open-output":
            opened, message = self.server.app.open_output_directory()
            status = HTTPStatus.OK if opened else HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_json({"ok": opened, "message": message}, status)
        else:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)


def _existing_server_url(server_file: Path) -> str | None:
    if not server_file.is_file():
        return None
    try:
        data = json.loads(server_file.read_text(encoding="utf-8"))
        port = int(data["port"])
        url = f"http://{DEFAULT_HOST}:{port}"
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as response:
            health = json.loads(response.read().decode("utf-8"))
        if health.get("app_id") == APP_ID:
            return url
    except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        return None
    return None


def _bind_server(app: FuxuanWebApplication, preferred_port: int) -> WebUIHTTPServer:
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + PORT_ATTEMPTS):
        try:
            return WebUIHTTPServer((DEFAULT_HOST, port), app)
        except OSError as error:
            last_error = error
    raise OSError(
        f"无法绑定本地端口 {preferred_port}-{preferred_port + PORT_ATTEMPTS - 1}"
    ) from last_error


def _show_startup_error(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "TTS 启动失败", 0x10)
    except (AttributeError, OSError):
        pass


def run_webui(*, preferred_port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    existing_url = _existing_server_url(SERVER_FILE)
    if existing_url is not None:
        if open_browser:
            webbrowser.open_new_tab(existing_url)
        return 0

    try:
        registry = load_voice_registry()
    except (OSError, ValueError) as error:
        _show_startup_error(f"音色模型注册表无效：{error}")
        return 1
    for directory in registry.input_dirs() + registry.output_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    store = StateStore(STATE_FILE, HISTORY_FILE)
    recover_and_start_session(
        store,
        state_dir=STATE_DIR,
        output_dir=registry.output_dirs(),
        default_model_id=registry.default_model_id,
    )
    app = FuxuanWebApplication(store, registry=registry)
    try:
        server = _bind_server(app, preferred_port)
    except OSError as error:
        store.transition("failed", f"WebUI 启动失败：{error}", errors=[str(error)])
        _show_startup_error(str(error))
        return 1

    app.server = server
    port = int(server.server_address[1])
    url = f"http://{DEFAULT_HOST}:{port}"
    SERVER_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVER_FILE.write_text(
        json.dumps(
            {"app_id": APP_ID, "pid": os.getpid(), "port": port, "url": url},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    watchdog = threading.Thread(
        target=app.watchdog,
        name="voice-webui-watchdog",
        daemon=True,
    )
    watchdog.start()
    if open_browser:
        threading.Timer(0.35, webbrowser.open_new_tab, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        if SERVER_FILE.is_file():
            try:
                registered = json.loads(SERVER_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                registered = {}
            if registered.get("pid") == os.getpid():
                SERVER_FILE.unlink(missing_ok=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local multi-voice TTS WebUI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_webui(preferred_port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
