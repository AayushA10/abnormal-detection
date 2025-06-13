import os
from pathlib import Path
from PIL import Image

# ---- CONFIG ----
INPUT_DIR  = "data/raw/UCSDped1/Test"      # change to Train / Test as needed
OUTPUT_DIR = "data/processed/UCSDped1/Test"
FRAME_SIZE = (227, 227)

def extract_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for clip in sorted(os.listdir(input_dir)):
        in_clip  = os.path.join(input_dir, clip)
        out_clip = os.path.join(output_dir, clip)
        if not os.path.isdir(in_clip):
            continue
        os.makedirs(out_clip, exist_ok=True)

        for idx, img_name in enumerate(sorted(os.listdir(in_clip))):
            src = os.path.join(in_clip, img_name)
            dst = os.path.join(out_clip, f"{idx:04d}.jpg")

            img = Image.open(src).resize(FRAME_SIZE)
            img.save(dst)
        print(f"✅  {clip}: {idx+1} frames")

if __name__ == "__main__":
    extract_frames(INPUT_DIR, OUTPUT_DIR)
