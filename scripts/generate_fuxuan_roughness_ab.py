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
    LEGACY_FUXUAN_VOICE_POLISH_CONFIG_PATH,
    apply_fuxuan_voice_polish,
    load_edge_trim_config,
    load_voice_polish_config,
)

DEFAULT_INPUT = ROOT / "inputs/fuxuan_text/新建 文本文档.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/fuxuan_roughness_ab"
V2_CONFIG = ROOT / "configs/lab/postprocess/fuxuan_voice_polish_roughness_control_v2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate matched Fuxuan post-processing A/B files from one synthesis pass."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--pause-ms", type=int, default=250)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    segments = split_text(input_path.read_text(encoding="utf-8-sig"))
    if not segments:
        raise ValueError(f"Input contains no speakable text: {input_path}")

    print(json.dumps({"stage": "loading_model"}, ensure_ascii=False), flush=True)
    runtime = load_runtime()
    print(
        json.dumps({"stage": "generating", "segments": len(segments)}, ensure_ascii=False),
        flush=True,
    )
    raw, sample_rate = render_segments(
        runtime,
        segments,
        edge_trim_config=load_edge_trim_config(DEFAULT_EDGE_TRIM_CONFIG),
        base_seed=args.seed,
        pause_ms=args.pause_ms,
    )

    variants = [
        ("A_current_nearfield_v1.wav", LEGACY_FUXUAN_VOICE_POLISH_CONFIG_PATH),
        ("B_roughness_control_v2.wav", V2_CONFIG),
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "input": str(input_path),
        "seed": args.seed,
        "segments": segments,
        "sample_rate": sample_rate,
        "variants": [],
    }
    for filename, config_path in variants:
        config = load_voice_polish_config(config_path)
        output, details = apply_fuxuan_voice_polish(raw, sample_rate, config)
        destination = output_dir / filename
        write_final_audio(destination, output, sample_rate)
        report["variants"].append(
            {
                "path": str(destination),
                "config": str(config_path),
                "duration_seconds": output.size / sample_rate,
                "processing": details,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": "completed", "output_dir": str(output_dir), "report": str(report_path)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
