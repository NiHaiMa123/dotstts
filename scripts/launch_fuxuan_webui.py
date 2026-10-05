from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def report_startup_error(error: BaseException) -> None:
    log_dir = ROOT / "outputs/fuxuan_webui"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "startup-error.log"
    details = "".join(traceback.format_exception(error))
    log_path.write_text(details, encoding="utf-8")
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            f"WebUI 未能启动。错误记录：\n{log_path}\n\n{error}",
            "TTS 启动失败",
            0x10,
        )
    except (AttributeError, OSError):
        pass


def run() -> int:
    try:
        from dots_tts_lab.fuxuan_webui import main

        return main()
    except BaseException as error:
        report_startup_error(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
