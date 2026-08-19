"""
Merge all recorded event clips into one summary video (ffmpeg stream-copy).
Run after stopping surveillance:
    .venv\Scripts\python.exe merge_events.py

The web dashboard (dashboard.py → Merge button) does the same thing.
Use either — this script is for command-line convenience.
"""

import os
import glob
import time
import subprocess

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(BASE_DIR, "events")


def _ensure_h264(filepath):
    marker = filepath + ".h264ok"
    if os.path.exists(marker):
        return
    tmp = filepath + ".tmp.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-y",
             "-i", filepath,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart",
             tmp],
            check=True, timeout=300
        )
        os.replace(tmp, filepath)
        open(marker, "w").close()
        print(f"  Converted {os.path.basename(filepath)} → H.264")
    except Exception as e:
        print(f"  [WARN] H.264 conversion failed for {os.path.basename(filepath)}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


def merge():
    files = sorted(
        glob.glob(os.path.join(EVENTS_DIR, "event_*.mp4")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )

    if not files:
        print("[INFO] No event files found in 'events/' folder.")
        print("       Run main.py first to record some events.")
        return

    print(f"[INFO] Found {len(files)} event(s) to merge.\n")
    for f in files:
        print(f"  • {os.path.basename(f)}")

    print("\n[INFO] Ensuring all clips are H.264...")
    for fp in files:
        _ensure_h264(fp)

    ts         = time.strftime("%Y%m%d_%H%M%S")
    out_path   = os.path.join(EVENTS_DIR, f"merged_{ts}.mp4")
    concat_txt = out_path + ".txt"

    with open(concat_txt, "w", encoding="utf-8") as f:
        for fp in files:
            f.write(f"file '{fp.replace(chr(92), '/')}'\n")

    print("\n[INFO] Merging... (stream copy — fast, no re-encode)")

    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-y",
             "-f", "concat", "-safe", "0",
             "-i", concat_txt,
             "-c", "copy",
             "-movflags", "+faststart",
             out_path],
            check=True, timeout=600
        )
        open(out_path + ".h264ok", "w").close()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg merge failed: {e}")
        return
    finally:
        if os.path.exists(concat_txt):
            os.remove(concat_txt)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[DONE] Saved    : {out_path}")
    print(f"[DONE] Events   : {len(files)}")
    print(f"[DONE] Size     : {size_mb:.1f} MB")
    print(f"\n[INFO] Open http://localhost:5000 to view in dashboard.")


if __name__ == "__main__":
    merge()
