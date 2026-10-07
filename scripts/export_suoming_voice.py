#!/usr/bin/env python3
"""Export 锁暝 (Suoming) voice lines from local game data to data/inbox.

Structure mirrors export_shouanren_voice.py but adapted to the newer
ConfigDB snapshot (extracted fresh to E:\\project\\temp\\suoming_cfg):

  - Favor voice lines: db_favor.db favorword RoleId=1312
    -> play_favor_word_suoming_* events -> .bnk in suoming_ext Event dir
    -> media id inside bnk -> .wem extracted on demand via FModelCLI
    (numeric media names need per-id extraction; each run ~2s).
  - Story/dialog lines: db_flowState.db flowstate BinData embeds an Actions
    JSON payload; TalkItems with WhoId in SPEAKER_WHO_IDS give TidTalk text
    keys -> zh_vo_<TidTalk>.wem in WwiseExternalSource (extracted per key).

Audio decoded with vgmstream-cli to
  data/inbox/锁暝/<emotion>/【<emotion>】<verbatim text>.wav
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
SUOMING_CFG = Path(r"E:\project\temp\suoming_cfg\Client\Content\Aki\ConfigDB")
SUOMING_EXT = Path(r"E:\project\temp\suoming_ext")
GAME_DIR = Path(r"D:\game\Wuthering Waves\Wuthering Waves Game")
AES_KEY = "0x6F80948821CA338739A24D4D9F778BCAC0996B2EF2A73897A789C68AFF05174E"
MEDIA_OUT = Path(r"E:\project\temp\suoming_media")
KNOWN_MEDIA_IDS = Path(r"E:\project\temp\known_media_ids.txt")

FMODELCLI = LUDIGLOT / "tools" / "FModelCLI.exe"
VGMSTREAM = LUDIGLOT / "tools" / ".data" / "vgmstream-cli.exe"

LANG_DBS = sorted((SUOMING_CFG / "zh-Hans").glob("*.db"))
ROLE_IDS = {1312}
SPEAKER_WHO_IDS = {400073, 400074, 7800, 8106}
VO_OUT = Path(r"E:\project\temp\suoming_vo")
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
                    if isinstance(key, str) and isinstance(value, str) and value:
                        textmap.setdefault(key, value)
        con.close()
    return textmap


def collect_favor_lines(textmap: dict[str, str]) -> list[dict]:
    con = sqlite3.connect(f"file:{SUOMING_CFG / 'db_favor.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT Id, RoleId, Type, Sort, BinData FROM favorword WHERE RoleId IN ({}) ORDER BY Sort".format(
            ",".join(map(str, ROLE_IDS))
        )
    ).fetchall()
    con.close()
    lines = []
    for fid, rid, ftype, sort, blob in rows:
        strings = [s.decode() for s in ASCII_RE.findall(blob)]
        events: list[str] = []
        for s in strings:
            events.extend(EVENT_RE.findall(s))
        events = [e for e in dict.fromkeys(events) if e.startswith("play_")]
        # 只保留 suoming 自己的事件（favor 行里可能混入 about_ 他人事件）
        events = [e for e in events if "_suoming_" in e or e.endswith("_suoming")]
        text_key = f"FavorWord_{rid * 100 + fid % 100}_Content"
        lines.append(
            {
                "kind": "favor",
                "role_id": rid,
                "text_key": text_key,
                "events": events,
                "text": textmap.get(text_key, ""),
                "favor_id": fid,
                "sort": sort,
            }
        )
    return lines


def collect_dialog_keys() -> list[str]:
    """db_flowState BinData 尾部嵌 JSON Actions；解析 TalkItems 取锁暝台词键。"""
    con = sqlite3.connect(f"file:{SUOMING_CFG / 'db_flowState.db'}?mode=ro", uri=True)
    keys: list[str] = []
    for (_k, blob) in con.execute("SELECT StateKey, BinData FROM flowstate"):
        i = blob.find(b"[")
        if i < 0:
            continue
        try:
            actions = json.loads(blob[i:])
        except Exception:
            j = blob.rfind(b"]")
            try:
                actions = json.loads(blob[i : j + 1])
            except Exception:
                continue
        for act in actions:
            for it in (act.get("Params") or {}).get("TalkItems") or []:
                if it.get("WhoId") in SPEAKER_WHO_IDS and it.get("TidTalk"):
                    keys.append(it["TidTalk"])
    con.close()
    return list(dict.fromkeys(keys))


def extract_vo_wem(text_key: str) -> Path | None:
    dst = (
        VO_OUT
        / "Client"
        / "Content"
        / "Aki"
        / "WwiseAudio_Generated"
        / "WwiseExternalSource"
        / f"zh_vo_{text_key}.wem"
    )
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    subprocess.run(
        [str(FMODELCLI), str(GAME_DIR), AES_KEY, str(VO_OUT), f"zh_vo_{text_key}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    return None


def media_ids_in_bnk(bnk_path: Path, known_media: set[int]) -> list[int]:
    data = bnk_path.read_bytes()
    found: list[int] = []
    for off in range(0, len(data) - 4):
        v = struct.unpack_from("<I", data, off)[0]
        if v in known_media and v not in found:
            found.append(v)
    return found


def extract_wem(media_id: int) -> Path | None:
    """FModelCLI 按 id 过滤提取单个 wem 到 MEDIA_OUT。每次约 2s。"""
    dst = MEDIA_OUT / "Client" / "Content" / "Aki" / "WwiseAudio_Generated" / "Media" / f"{media_id}.wem"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    result = subprocess.run(
        [str(FMODELCLI), str(GAME_DIR), AES_KEY, str(MEDIA_OUT), str(media_id)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    return None


def clean_text(text: str) -> str:
    text = TE_TAG.sub("", text)
    text = TAG.sub("", text)
    text = PLAYER_PLACEHOLDERS.sub("{TA}", text)
    return text.strip()


def filename_for(text: str) -> str:
    stem = INVALID_CHARS.sub("", clean_text(text))
    stem = re.sub(r"\s+", "", stem)
    stem = stem[:80]
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
        default=Path(r"E:\project\dotstts\data\inbox\锁暝") / EMOTION,
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    textmap = load_textmap()
    items = collect_favor_lines(textmap)
    print(f"favor lines: {len(items)}")

    dialog_keys = collect_dialog_keys()
    print(f"dialog keys: {len(dialog_keys)}")
    for key in dialog_keys:
        items.append(
            {
                "kind": "dialog",
                "text_key": key,
                "events": [f"zh_vo_{key}"],
                "text": textmap.get(key, ""),
            }
        )

    known_media = {
        int(line) for line in KNOWN_MEDIA_IDS.read_text().split() if line.strip()
    }
    print(f"known media ids: {len(known_media)}")

    # 预收集全部 media id：bnk 里的数字候选先记下来，提取时验证
    missing_text = [i for i in items if not i["text"]]
    missing_event = [i for i in items if not i["events"]]
    print(f"missing_text={len(missing_text)}  missing_event={len(missing_event)}")

    used_names: dict[str, int] = {}
    exported: list[dict] = []
    for item in items:
        if not item["text"]:
            continue
        wem: Path | None = None
        if item["kind"] == "dialog":
            wem = extract_vo_wem(item["text_key"])
        else:
            for event in item["events"]:
                bnk = SUOMING_EXT / "Client" / "Content" / "Aki" / "WwiseAudio_Generated" / "Event" / f"{event}.bnk"
                if not bnk.exists():
                    continue
                for mid in media_ids_in_bnk(bnk, known_media):
                    cand = extract_wem(mid)
                    if cand:
                        wem = cand
                        break
                if wem:
                    break
        base = filename_for(item["text"])
        if base in used_names:
            used_names[base] += 1
            base = f"{base[:-4]}_{used_names[base]}.wav"
        else:
            used_names[base] = 1
        out_path = args.out_dir / base
        ok = True
        if not args.dry_run:
            ok = bool(wem) and decode_wem(wem, out_path)
        exported.append(
            {
                "text_key": item["text_key"],
                "kind": item["kind"],
                "events": item["events"],
                "wem": str(wem) if wem else None,
                "out": str(out_path),
                "text": clean_text(item["text"]),
                "ok": ok,
            }
        )
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

    failed = [e for e in exported if not e["ok"]]
    report = {
        "role": "锁暝",
        "role_ids": sorted(ROLE_IDS),
        "out_dir": str(args.out_dir),
        "exported": len(exported) - len(failed),
        "failed": len(failed),
        "missing_text": [i["text_key"] for i in missing_text],
        "missing_event": [i["text_key"] for i in missing_event],
        "items": exported,
    }
    report_path = args.report or Path(r"E:\project\temp\suoming_export_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported {report['exported']} / {len(exported)}  failed={len(failed)}  report={report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
