import os, io, pandas as pd, streamlit as st
from PIL import Image

# ===== CONFIG =====
DATA_DIR    = "data/processed/UCSDped1/Test"
CSV_PATH    = "outputs/frame_errors.csv"
FRAME_SIZE  = (227, 227)
VIDEO_PATH  = "outputs/anomaly_video.mp4"

# ===== Load data =====
st.set_page_config(page_title="Abnormal Detection", layout="centered")
st.title("🧠 Abnormal Event Detection – Dashboard")

df = pd.read_csv(CSV_PATH)
default_thr = float(df["mse"].mean() + 2.5 * df["mse"].std())   # loose threshold
thr = st.slider("Anomaly threshold", 0.0, 0.01, default_thr, 0.0001)
df["is_anomaly"] = df["mse"] > thr

# ===== Plot =====
st.subheader("📉 Reconstruction Error")
st.line_chart(df["mse"])

# ===== Anomaly tables & downloads =====
anom_df  = df[df["is_anomaly"]].copy()

top10_df = anom_df.sort_values("mse", ascending=False).head(10)

with st.expander("📊  Top-10 Anomalies (highest MSE)"):
    st.dataframe(top10_df.style.format({"mse": "{:.5f}"}), height=300)

    # --- Download buttons ---
    def df_to_bytes(dframe):
        buf = io.StringIO()
        dframe.to_csv(buf, index=False)
        return buf.getvalue().encode()

    colA, colB = st.columns(2)
    colA.download_button("⬇️ Download TOP-10 CSV",
                         data=df_to_bytes(top10_df),
                         file_name="top10_anomalies.csv",
                         mime="text/csv")
    colB.download_button("⬇️ Download ALL Anomalies CSV",
                         data=df_to_bytes(anom_df),
                         file_name="all_anomalies.csv",
                         mime="text/csv")

# ===== Frame viewer =====
st.subheader("🖼️  Frame Viewer")

show_only_anom = st.checkbox("Show only anomalies", value=True)
view_df = anom_df if show_only_anom else df
if view_df.empty:
    st.warning("No frames match current threshold.")
    st.stop()

idx = st.slider("Frame index", 0, len(view_df) - 1, 0)
row = view_df.iloc[idx]
fidx = int(row["frame_idx"])

def resolve_path(idx: int):
    for clip in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, clip, f"{idx:04d}.jpg")
        if os.path.exists(p):
            return p
    return None

img_path = resolve_path(fidx)
if img_path:
    img = Image.open(img_path).resize(FRAME_SIZE)
    st.image(img, caption=f"Frame {fidx} • MSE={row['mse']:.5f} • "
                          f"{'🚨 Anomaly' if row['is_anomaly'] else '✅ Normal'}",
             use_column_width=True)
else:
    st.warning(f"Frame {fidx} not found in processed folder.")

# ===== Video download =====
if os.path.exists(VIDEO_PATH):
    with open(VIDEO_PATH, "rb") as video_file:
        st.download_button("🎞️  Download Anomaly Video",
                           data=video_file,
                           file_name="anomaly_video.mp4",
                           mime="video/mp4")
