#!/usr/bin/env python3
"""Export 守岸人 (Shorekeeper) voice lines from local game data to data/inbox.

Sources:
  - Favor voice lines: db_favor.db (favorword, RoleId=1505) -> play_favor_word_* .bnk
    -> media id inside bnk -> .wem inside role_lang_shouanren_zh pak extraction.
  - Story/dialog lines: FlowState.json TalkItems with WhoId in
    {1398, 701100, 750019} (Speaker_*_Name == 守岸人) -> zh_vo_<TextKey>.wem
    inside the already-extracted WwiseExternalSource directory.
  - voice_map extras (LevelPlay / Gameplay keys) are merged in when they carry
    a resolvable zh_vo_* external source.

Audio is decoded with vgmstream-cli and written as
  data/inbox/守岸人/<emotion>/【<emotion>】<verbatim text>.wav
matching configs/lab/metadata_profiles/speaker_emotion_filename_v1.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

LUDIGLOT = Path(r"E:\project\Ludiglot")
WUTHERING_DATA = Path(r"E:\project\WutheringData")
ROLE_PAK_MEDIA = Path(
    r"E:\project\temp\shouanren_pak\Client\Content\Aki\WwiseAudio_Generated\Media"
)
MAIN_EVENT = Path(
    r"E:\project\temp\main_event\Client\Content\Aki\WwiseAudio_Generated\Event"
)
EXTERNAL_SOURCE = LUDIGLOT / "data" / "WwiseAudio_Generated" / "WwiseExternalSource"
VGMSTREAM = LUDIGLOT / "tools" / ".data" / "vgmstream-cli.exe"

CONFIGDB = LUDIGLOT / "data" / "ConfigDB"
LANG_DBS = [
    CONFIGDB / "zh-Hans" / "lang_multi_text.db",
    CONFIGDB / "zh-Hans" / "lang_multi_text_1sthalf.db",
    CONFIGDB / "zh-Hans" / "lang_flow_text.db",
    CONFIGDB / "zh-Hans" / "lang_subtitle_text.db",
    CONFIGDB / "zh-Hans" / "lang_chat.db",
]

ROLE_ID = 1505
SPEAKER_WHO_IDS = {1398, 701100, 750019}
EMOTION = "中立_neutral"
INVALID_CHARS = re.compile(r'[\\/:*?"<>|]')
TE_TAG = re.compile(r"</?te[^>]*>")
TAG = re.compile(r"<[^>]+>")
PLAYER_PLACEHOLDERS = re.compile(r"\{PlayerName\}|\{NICKNAME\}|\{playerName\}|\{Player\}")
ASCII_RE = re.compile(rb"[\x20-\x7e]{4,}")
EVENT_RE = re.compile(r"(?:play|vo)_[a-zA-Z0-9_]+")


def load_textmap() -> dict[str, str]:
    textmap: dict[str, str] = {}
    for db in LANG_DBS:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            cols = [c[1] for c in con.execute(f"PRAGMA table_info({table})")]
            if "Id" in cols and "Content" in cols:
                for key, value in con.execute(f"SELECT Id, Content FROM {table}"):
                    if isinstance(key, str) and isinstance(value, str):
                        textmap.setdefault(key, value)
        con.close()
    return textmap


def load_plot_audio() -> dict[str, str]:
    payload = json.loads(
        (WUTHERING_DATA / "ConfigDB" / "PlotAudio.json").read_text(encoding="utf-8")
    )
    return {
        item["Id"]: item["FileName"]
        for item in payload
        if item.get("Id") and item.get("FileName")
    }


def collect_favor_lines(textmap: dict[str, str]) -> list[dict]:
    con = sqlite3.connect(f"file:{CONFIGDB / 'db_favor.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT Id, Type, Sort, BinData FROM favorword WHERE RoleId=? ORDER BY Sort",
        (ROLE_ID,),
    ).fetchall()
    con.close()
    lines = []
    for fid, ftype, sort, blob in rows:
        strings = [s.decode() for s in ASCII_RE.findall(blob)]
        events: list[str] = []
        for s in strings:
            events.extend(EVENT_RE.findall(s))
        events = [e for e in dict.fromkeys(events) if e.startswith("play_")]
        text_key = f"FavorWord_{fid}_Content"
        lines.append(
            {
                "kind": "favor",
                "text_key": text_key,
                "events": events,
                "text": textmap.get(text_key, ""),
                "favor_id": fid,
                "sort": sort,
            }
        )
    return lines


def collect_dialog_lines() -> list[str]:
    flows = json.loads(
        (WUTHERING_DATA / "ConfigDB" / "FlowState.json").read_text(encoding="utf-8")
    )
    keys: list[str] = []
    for flow in flows:
        try:
            actions = json.loads(flow.get("Actions") or "[]")
        except Exception:
            continue
        for action in actions:
            params = action.get("Params") or {}
            for item in params.get("TalkItems") or []:
                if item.get("WhoId") in SPEAKER_WHO_IDS and item.get("TidTalk"):
                    keys.append(item["TidTalk"])
    return list(dict.fromkeys(keys))


def media_ids_in_bnk(bnk_path: Path, known_media: set[int]) -> list[int]:
    data = bnk_path.read_bytes()
    found: list[int] = []
    for offset in range(0, len(data) - 4):
        value = struct.unpack_from("<I", data, offset)[0]
        if value in known_media and value not in found:
            found.append(value)
    return found


def clean_text(text: str) -> str:
    text = TE_TAG.sub("", text)
    text = TAG.sub("", text)
    text = PLAYER_PLACEHOLDERS.sub("{TA}", text)
    return text.strip()


def filename_for(text: str) -> str:
    stem = INVALID_CHARS.sub("", clean_text(text))
    return f"【{EMOTION}】{stem}.wav"


def decode_wem(wem: Path, out_wav: Path) -> bool:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists() and out_wav.stat().st_size > 44:
        return True
    result = subprocess.run(
        [str(VGMSTREAM), "-o", str(out_wav), str(wem)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and out_wav.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"E:\project\dotstts\data\inbox\守岸人") / EMOTION,
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    textmap = load_textmap()
    plot_audio = load_plot_audio()

    known_media: dict[int, Path] = {}
    for wem in ROLE_PAK_MEDIA.rglob("*.wem"):
        if wem.stem.isdigit():
            known_media[int(wem.stem)] = wem
    print(f"role pak media wems: {len(known_media)}")

    items: list[dict] = []

    # --- favor lines -----------------------------------------------------
    for line in collect_favor_lines(textmap):
        wems: list[Path] = []
        for event in line["events"]:
            bnk = MAIN_EVENT / f"{event}.bnk"
            if not bnk.exists():
                continue
            for media_id in media_ids_in_bnk(bnk, set(known_media)):
                wems.append(known_media[media_id])
        items.append({**line, "wems": wems})

    # --- dialog lines -----------------------------------------------------
    for key in collect_dialog_lines():
        event = plot_audio.get(key, "")
        candidates = []
        if event:
            candidates.append(EXTERNAL_SOURCE / f"zh_{event}.wem")
        candidates.extend(
            [
                EXTERNAL_SOURCE / f"zh_vo_{key}.wem",
                EXTERNAL_SOURCE / f"zh_play_vo_{key}.wem",
            ]
        )
        wem = next((p for p in candidates if p.exists()), None)
        items.append(
            {
                "kind": "dialog",
                "text_key": key,
                "events": [event],
                "text": textmap.get(key, ""),
                "wems": [wem] if wem else [],
            }
        )

    missing_text = [i for i in items if not i["text"]]
    missing_audio = [i for i in items if not i["wems"]]
    print(f"items: {len(items)}  missing_text: {len(missing_text)}  missing_audio: {len(missing_audio)}")

    used_names: dict[str, int] = {}
    exported: list[dict] = []
    for item in items:
        if not item["text"] or not item["wems"]:
            continue
        base = filename_for(item["text"])
        if base in used_names:
            used_names[base] += 1
            stem = base[: -len(".wav")]
            base = f"{stem}_{used_names[base]}.wav"
        else:
            used_names[base] = 1
        out_path = args.out_dir / base
        ok = True
        if not args.dry_run:
            ok = decode_wem(item["wems"][0], out_path)
            if not ok and len(item["wems"]) > 1:
                ok = decode_wem(item["wems"][1], out_path)
        exported.append(
            {
                "text_key": item["text_key"],
                "kind": item["kind"],
                "events": item["events"],
                "wems": [str(w) for w in item["wems"]],
                "out": str(out_path),
                "text": clean_text(item["text"]),
                "ok": ok,
            }
        )

    failed = [e for e in exported if not e["ok"]]
    report = {
        "role": "守岸人",
        "out_dir": str(args.out_dir),
        "exported": len(exported) - len(failed),
        "failed": len(failed),
        "missing_text": [i["text_key"] for i in missing_text],
        "missing_audio": [i["text_key"] for i in missing_audio],
        "items": exported,
    }
    report_path = args.report or Path(r"E:\project\temp\shouanren_export_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"exported {report['exported']} / {len(exported)}  failed={len(failed)}  "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
