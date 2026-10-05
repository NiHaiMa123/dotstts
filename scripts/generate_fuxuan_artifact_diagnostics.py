from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dots_tts_lab.fuxuan_batch import (  # noqa: E402
    DEFAULT_EDGE_TRIM_CONFIG,
    load_runtime,
    render_segments,
    split_text,
    write_final_audio,
)
from dots_tts_lab.postprocess import (  # noqa: E402
    apply_fuxuan_voice_polish,
    load_edge_trim_config,
    load_voice_polish_config,
)

DEFAULT_INPUT = ROOT / "inputs/fuxuan_text/新建 文本文档.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/fuxuan_roughness_ab"
V2_CONFIG = ROOT / "configs/lab/postprocess/fuxuan_voice_polish_roughness_control_v2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate matched candidates that isolate step-count and seed artifacts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    segments = split_text(input_path.read_text(encoding="utf-8-sig"))
    edge_trim = load_edge_trim_config(DEFAULT_EDGE_TRIM_CONFIG)
    polish = load_voice_polish_config(V2_CONFIG)
    candidates = [
        {
            "filename": "C_same_seed_steps16.wav",
            "seed": 20260908,
            "num_steps": 16,
            "purpose": "isolate increased generation steps",
        },
        {
            "filename": "D_alt_seed_steps10.wav",
            "seed": 20260918,
            "num_steps": 10,
            "purpose": "isolate alternate stochastic generation",
        },
    ]

    print(json.dumps({"stage": "loading_model"}), flush=True)
    runtime = load_runtime()
    report: dict[str, object] = {
        "schema_version": 1,
        "input": str(input_path),
        "segments": segments,
        "postprocess_config": str(V2_CONFIG),
        "candidates": [],
    }
    for index, candidate in enumerate(candidates, start=1):
        print(
            json.dumps(
                {
                    "stage": "generating_candidate",
                    "index": index,
                    "count": len(candidates),
                    "seed": candidate["seed"],
                    "num_steps": candidate["num_steps"],
                }
            ),
            flush=True,
        )
        raw, sample_rate = render_segments(
            runtime,
            segments,
            edge_trim_config=edge_trim,
            base_seed=int(candidate["seed"]),
            pause_ms=250,
            num_steps=int(candidate["num_steps"]),
        )
        output, processing = apply_fuxuan_voice_polish(raw, sample_rate, polish)
        destination = output_dir / str(candidate["filename"])
        write_final_audio(destination, output, sample_rate)
        report["candidates"].append(
            {
                **candidate,
                "path": str(destination),
                "sample_rate": sample_rate,
                "duration_seconds": output.size / sample_rate,
                "processing": processing,
            }
        )

    report_path = output_dir / "artifact_diagnostics_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"stage": "completed", "report": str(report_path)}, ensure_ascii=False
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
