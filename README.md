# 🎥 UCSD Abnormal Detection System with Streamlit Dashboard

An end-to-end anomaly detection system using a convolutional autoencoder trained on the UCSD Ped1 dataset. The pipeline includes preprocessing, model training, anomaly scoring, visualization, and an interactive Streamlit dashboard.

## 📂 Project Structure

├── data/ # Raw & processed UCSD dataset
│ ├── raw/
│ └── processed/
├── models/ # Trained model (.pth)
├── outputs/ # Evaluation CSV, plots, and heatmaps (gitignored)
├── scripts/
│ ├── extract_frames.py
│ ├── preprocess_data.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── generate_video.py
│ └── visualize_results.py
├── utils/ # Helper functions
├── streamlit_appp.py # 📊 Streamlit Dashboard
├── requirements.txt
├── README.md
└── .gitignore


---

## 🚀 Features

✅ **Autoencoder-based anomaly detection**  
✅ **Frame-wise MSE scoring**  
✅ **Reconstruction error plots**  
✅ **Top-10 anomaly frame viewer**  
✅ **Interactive Streamlit dashboard**  
✅ **Video generation with anomaly overlays**

---

## 📦 Installation

```bash
git clone https://github.com/AayushA10/UCSD-Abnormal-Detection-System-with-Streamlit-dashboard.git
cd UCSD-Abnormal-Detection-System-with-Streamlit-dashboard
pip install -r requirements.txt

🛠️ Steps to Run
1️⃣ Extract frames from video
python3 scripts/extract_frames.py
2️⃣ Train the autoencoder
python3 scripts/train_model.py
3️⃣ Evaluate on test data
python3 scripts/evaluate_model.py
4️⃣ Launch the Streamlit Dashboard
streamlit run streamlit_appp.py
📊 Dashboard Features
📈 Reconstruction error plot

🎯 Adjustable anomaly threshold

🔍 Frame viewer with MSE & prediction

🏆 Auto-highlight Top-10 anomalous frames

⬇️ Download anomaly visualization video

📤 Export anomaly frames as CSV

🧠 Model Details
Convolutional Autoencoder (CNN)

Trained on grayscale resized (227x227) frames

Reconstruction error (MSE) used for anomaly detection

Threshold set via mean + k * std
