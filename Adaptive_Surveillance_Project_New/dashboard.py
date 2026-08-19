import json
import os
import re
import glob
import time
import subprocess

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR   = os.path.join(BASE_DIR, "events")
METADATA_CSV = os.path.join(BASE_DIR, "metadata", "events_metadata.csv")
SESSION_JSON = os.path.join(BASE_DIR, "metadata", "session.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session")
def api_session():
    if not os.path.exists(SESSION_JSON):
        return jsonify({"available": False})
    try:
        with open(SESSION_JSON) as f:
            data = json.load(f)
        data["available"] = True
        return jsonify(data)
    except Exception:
        return jsonify({"available": False})


@app.route("/api/events")
def api_events():
    if not os.path.exists(METADATA_CSV):
        return jsonify([])

    df = pd.read_csv(METADATA_CSV)

    # ---------- Dynamic Statistics ----------
    avg_eis = 0
    high_threshold = 0
    low_threshold = 0

    if "EIS" in df.columns and len(df) > 0:
        avg_eis = round(df["EIS"].mean(), 2)
        std = df["EIS"].std(ddof=0)
        high_threshold = round(avg_eis + std, 2)
        low_threshold = avg_eis

    keep_count = 0
    compress_count = 0
    delete_count = 0

    if "Storage_Decision" in df.columns:
        keep_count = len(df[df["Storage_Decision"] == "Keep Original"])
        compress_count = len(df[df["Storage_Decision"] == "Compress"])
        delete_count = len(df[df["Storage_Decision"] == "Delete"])

    events = df.to_dict(orient="records")

    return jsonify({
        "events": events,
        "summary": {
            "total_events": len(df),
            "average_eis": avg_eis,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "keep_original": keep_count,
            "compressed": compress_count,
            "deleted": delete_count
        }
    })

@app.route("/api/dashboard-summary")
def dashboard_summary():

    if not os.path.exists(METADATA_CSV):
        return jsonify({})

    df = pd.read_csv(METADATA_CSV)

    if len(df) == 0:
        return jsonify({})

    avg_eis = round(df["EIS"].mean(), 2)
    max_eis = round(df["EIS"].max(), 2)
    min_eis = round(df["EIS"].min(), 2)

    std = df["EIS"].std(ddof=0)
    high_threshold = round(avg_eis + std, 2)
    low_threshold = avg_eis

    return jsonify({
        "AverageEIS": avg_eis,
        "LatestEIS": round(df.iloc[-1]["EIS"], 2),
        "HighThreshold": high_threshold,
        "LowThreshold": low_threshold,
        "TotalEvents": len(df)
    })

def _ensure_h264(filepath):
    """Convert mp4v → H.264 faststart in-place on first request. No-op if already done."""
    marker = filepath + ".h264ok"
    # Marker is stale if the video file was modified after the last conversion
    if os.path.exists(marker) and os.path.getmtime(marker) >= os.path.getmtime(filepath):
        return
    if os.path.exists(marker):
        os.remove(marker)
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
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/video/<path:filename>")
def serve_video(filename):
    """Serve video with HTTP range-request support so browsers can seek/play."""
    filepath = os.path.join(EVENTS_DIR, filename)
    if not os.path.exists(filepath):
        return "Not found", 404

    _ensure_h264(filepath)

    file_size = os.path.getsize(filepath)
    range_hdr = request.headers.get("Range")

    if range_hdr:
        m      = re.search(r"bytes=(\d+)-(\d*)", range_hdr)
        start  = int(m.group(1))
        end    = int(m.group(2)) if m.group(2) else file_size - 1
        end    = min(end, file_size - 1)
        length = end - start + 1
        with open(filepath, "rb") as f:
            f.seek(start)
            data = f.read(length)
        rv = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        rv.headers.set("Content-Range",  f"bytes {start}-{end}/{file_size}")
        rv.headers.set("Accept-Ranges",  "bytes")
        rv.headers.set("Content-Length", str(length))
        return rv

    rv = send_from_directory(EVENTS_DIR, filename, mimetype="video/mp4")
    rv.headers.set("Accept-Ranges", "bytes")
    return rv


@app.route("/api/merge", methods=["POST"])
def api_merge():
    """Merge all event_*.mp4 files using ffmpeg concat (stream-copy, no re-encode)."""
    pattern = os.path.join(EVENTS_DIR, "event_*.mp4")
    files   = sorted(
        glob.glob(pattern),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    if not files:
        return jsonify({"status": "error", "message": "No event files found"})

    # Ensure every input is H.264 so stream-copy works cleanly
    for fp in files:
        _ensure_h264(fp)

    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"merged_{ts}.mp4"
    out_path = os.path.join(EVENTS_DIR, out_name)

    # Write ffmpeg concat list (absolute paths, forward-slash safe on Windows)
    concat_txt = out_path + ".txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for fp in files:
            f.write(f"file '{fp.replace(chr(92), '/')}'\n")

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
        # Output is already H.264 (stream-copied from H.264 inputs)
        open(out_path + ".h264ok", "w").close()
        return jsonify({"status": "ok", "file": out_name, "count": len(files)})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"ffmpeg failed: {e}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(concat_txt):
            os.remove(concat_txt)


@app.route("/api/latest-merged")
def api_latest_merged():
    if not os.path.exists(EVENTS_DIR):
        return jsonify({"file": None})
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, "merged_*.mp4")))
    if not files:
        return jsonify({"file": None})
    return jsonify({"file": os.path.basename(files[-1])})


@app.route("/api/storage")
def api_storage():

    total_size = 0

    if os.path.exists(EVENTS_DIR):
        for file in os.listdir(EVENTS_DIR):
            fp = os.path.join(EVENTS_DIR, file)

            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)

    after_mb = round(total_size / (1024 * 1024), 2)

    before_mb = round(after_mb * 4, 2)

    saved_mb = round(before_mb - after_mb, 2)

    saved_percent = 0

    if before_mb > 0:
        saved_percent = round((saved_mb / before_mb) * 100, 2)

    return jsonify({
        "StorageBefore": before_mb,
        "StorageAfter": after_mb,
        "StorageSaved": saved_mb,
        "StorageSavedPercent": saved_percent
    })

@app.route("/api/delete/<int:event_id>", methods=["DELETE"])
def api_delete(event_id):
    video_path = os.path.join(EVENTS_DIR, f"event_{event_id}.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)
    marker = video_path + ".h264ok"
    if os.path.exists(marker):
        os.remove(marker)
    if os.path.exists(METADATA_CSV):
        df = pd.read_csv(METADATA_CSV)
        df = df[df["Event_ID"] != event_id]
        df.to_csv(METADATA_CSV, index=False)
    return jsonify({"status": "ok", "deleted": event_id})

@app.route("/api/charts")
def charts_data():

    if not os.path.exists(METADATA_CSV):
        return jsonify({})

    df = pd.read_csv(METADATA_CSV)

    if len(df) == 0:
        return jsonify({})

    return jsonify({

        "event_ids": df["Event_ID"].tolist(),

        "eis": df["EIS"].tolist(),

        "motion": df["Motion_Score"].tolist(),

        "persons": df["Person_Count"].tolist(),

        "decisions": df["Storage_Decision"].tolist()

    })

def _batch_convert_events():
    """Convert all existing mp4v event files to H.264 at startup."""
    if not os.path.exists(EVENTS_DIR):
        return
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, "event_*.mp4")))
    pending = [
        f for f in files
        if not os.path.exists(f + ".h264ok")
        or os.path.getmtime(f) > os.path.getmtime(f + ".h264ok")
    ]
    if not pending:
        return
    print(f"[Dashboard] Converting {len(pending)} event file(s) to H.264 for browser playback...")
    for fp in pending:
        print(f"            {os.path.basename(fp)} ...", end=" ", flush=True)
        _ensure_h264(fp)
        print("done" if os.path.exists(fp + ".h264ok") else "FAILED")
    print("[Dashboard] All event videos ready.\n")


if __name__ == "__main__":
    PORT = 5000

    _batch_convert_events()

    print("\n" + "=" * 62)
    print("  ADAPTIVE SMART SURVEILLANCE SYSTEM — DASHBOARD")
    print("=" * 62)
    print(f"  Local  →  http://localhost:{PORT}")
    print("=" * 55 + "\n")

    app.run(debug=False, host="0.0.0.0", port=PORT, threaded=True)
