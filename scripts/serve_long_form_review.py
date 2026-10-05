from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports" / "long_form"
INCOMING_DIR = REPORT_DIR / "incoming"

# Fixed landing paths: the page exports, the file lands here, no dialog.
# Names match ^[a-z0-9-]+$ and land under incoming/<name>.json.
SAVE_NAME_PATTERN = re.compile(r"^[a-z0-9-]{2,64}$")
INCOMING_PREFIXES = ("reference-", "strict-review-")


def _save_target(name: str) -> Path | None:
    if not SAVE_NAME_PATTERN.match(name) or not name.startswith(INCOMING_PREFIXES):
        return None
    return INCOMING_DIR / f"{name}.json"

PAGES = {
    "参考包确认（normal 12 条）": "data/reports/long_form/reference_pack_review_v1.html",
    "难负例确认（4 类各 6 条）": "data/reports/long_form/reference_negative_review_v1.html",
    "难负例补选·双耳（v2）": "data/reports/long_form/reference_negative_binaural_v2.html",
    "严格候选审核（LF-12H）": "data/reports/long_form/strict_review_46d41e37fc4f.html",
    "候选文本校订（86 条）": "data/reports/long_form/strict_text_46d41e37fc4f.html",
    "DFN3 对照验收 v3（12 句·8/12dB）": "data/reports/long_form/dfn3_ab_v3_46d41e37fc4f.html",
    "批量审核总览（旧）": "data/reports/long_form/review_index.html",
}


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[review] " + format % args + "\n")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_index()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if not self.path.startswith("/save/"):
            self.send_error(404)
            return
        name = self.path[len("/save/"):].split("?")[0]
        target = _save_target(name)
        if target is None:
            self.send_error(404, "unknown save target")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_error(400)
            return
        if not 0 < length <= 16 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(length)
        try:
            json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, "body must be valid JSON")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_bytes(body)
        os.replace(partial, target)
        sys.stderr.write(f"[review] saved {name} -> {target}\n")
        payload = json.dumps({"saved": str(target)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_index(self) -> None:
        cards = []
        for title, rel in PAGES.items():
            exists = (ROOT / rel).is_file()
            href = "/" + rel if exists else "#"
            note = "" if exists else "（尚未生成）"
            cards.append(
                f'<li><a href="{href}">{title}</a> <small>{note}</small></li>'
            )
        saved = []
        if INCOMING_DIR.is_dir():
            for path in sorted(INCOMING_DIR.glob("*.json")):
                mtime = time.strftime(
                    "%H:%M:%S", time.localtime(path.stat().st_mtime)
                )
                saved.append(f"<li>{path.name} · {path.stat().st_size}B · {mtime}</li>")
        saved_html = "".join(saved) or "<li>暂无已保存的导出</li>"
        page = (
            "<!doctype html><html lang=zh-CN><head><meta charset=utf-8>"
            "<title>长音频审核入口</title><style>body{font:15px system-ui;margin:40px;"
            "background:#f5f7fb;color:#172033}main{max-width:760px;margin:auto;background:white;"
            "border:1px solid #dce2ec;border-radius:12px;padding:24px}li{margin:10px 0}"
            "a{color:#2457d6}</style></head><body><main>"
            "<h1>长音频审核入口</h1><h2>页面</h2><ul>"
            + "".join(cards)
            + "</ul><h2>已收到的导出</h2><ul>"
            + saved_html
            + "</ul><p><small>导出直接写入 data/reports/long_form/incoming/，"
            "保存后告诉 Devin 即可。</small></p></main></body></html>"
        )
        payload = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve long-form review pages with fixed-path JSON export."
    )
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review server: {url}")
    print(f"exports land in: {INCOMING_DIR}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
