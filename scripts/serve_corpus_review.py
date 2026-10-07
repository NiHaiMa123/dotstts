#!/usr/bin/env python3
"""Static review server + save endpoint for corpus review pages.

Serves the repo root so review pages under ``outputs/<name>_review/`` can reach
``data/work/standardized/...`` audio, and accepts::

    POST /save_review/<name>   body = review decisions JSON

which writes two copies:

  - ``outputs/<name>_review/state.json`` — the live backup the page reloads on
    open (survives cleared localStorage)
  - ``data/reports/review_web/<name>_corpus_decisions.json`` — the pipeline-side
    record consumed when freezing the dataset

``<name>`` must be ``[a-z0-9_]+`` and ``outputs/<name>_review/`` must exist, so
the endpoint can never write outside review dirs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVE_RE = re.compile(r"^/save_review/([a-z0-9_]{2,64})$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[review] " + format % args + "\n")

    def do_POST(self) -> None:
        m = SAVE_RE.match(self.path.split("?", 1)[0])
        if not m:
            self.send_error(404)
            return
        name = m.group(1)
        review_dir = ROOT / "outputs" / f"{name}_review"
        if not review_dir.is_dir():
            self.send_error(404, "unknown review page")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_error(411)
            return
        if not 0 < length <= 10_000_000:
            self.send_error(413)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self.send_error(400, "invalid json")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), dict):
            self.send_error(400, "expected decisions object")
            return

        payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        (review_dir / "state.json").write_text(state, encoding="utf-8")
        out = ROOT / "data" / "reports" / "review_web" / f"{name}_corpus_decisions.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(state, encoding="utf-8")
        sys.stderr.write(f"[review] saved {name} -> {out} ({len(payload['decisions'])} decisions)\n")
        body = json.dumps({"saved": str(out)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {ROOT} at {url}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
