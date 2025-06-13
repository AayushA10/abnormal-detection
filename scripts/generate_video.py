"""
generate_video.py – creates side-by-side (original | heat-map) video
Adds red border + overlay text if frame is an anomaly.
"""

import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# -------- CONFIG --------
DATA_DIR    = "data/processed/UCSDped1/Test"
HEAT_DIR    = "outputs/heatmaps"
CSV_PATH    = "outputs/frame_errors.csv"
OUT_VIDEO   = "outputs/anomaly_video.mp4"
FPS         = 10

# -------- Load Data --------
df = pd.read_csv(CSV_PATH).sort_values("frame_idx")
writer = imageio.get_writer(OUT_VIDEO, fps=FPS)

# -------- Frame Resolver --------
def find_original(idx):
    for clip in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, clip, f"{idx:04d}.jpg")
        if os.path.exists(path):
            return path
    return None

# -------- Optional Font --------
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# -------- Process Frames --------
for _, row in df.iterrows():
    idx     = int(row["frame_idx"])
    mse     = row["mse"]
    is_anom = row["is_anomaly"]

    orig_p  = find_original(idx)
    heat_p  = os.path.join(HEAT_DIR, f"{idx:04d}.jpg")
    if not (orig_p and os.path.exists(heat_p)):
        continue

    orig  = Image.open(orig_p).convert("RGB")
    heat  = Image.open(heat_p).convert("RGB")

    combo = Image.new("RGB", (orig.width * 2, orig.height))
    combo.paste(orig, (0, 0))
    combo.paste(heat, (orig.width, 0))

    draw = ImageDraw.Draw(combo)

    # ---- Label Overlay ----
    label = f"Frame {idx:04d}  |  {'🚨 Anomaly' if is_anom else '✅ Normal'}  |  MSE={mse:.5f}"
    draw.rectangle([0, 0, combo.width, 28], fill=(0, 0, 0, 180))
    draw.text((10, 5), label, fill="white", font=font)

    # ---- Red Border if Anomaly ----
    if is_anom:
        draw.rectangle([0, 0, combo.width - 1, combo.height - 1], outline="red", width=4)

    writer.append_data(np.array(combo))

writer.close()
print(f"🎞️  Video saved → {OUT_VIDEO}")
