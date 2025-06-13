"""
evaluate_model.py
 • computes frame-wise MSE
 • saves reconstructed frames
 • saves heat-maps (abs(original – recon))
 • writes frame_errors.csv
"""
import os, csv, torch, torch.nn as nn
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image, ImageChops
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# -------- CONFIG --------
DATA_DIR   = "data/processed/UCSDped1/Test"
MODEL_PATH = "models/autoencoder.pth"
IMG_SIZE   = (227, 227)
CSV_PATH   = "outputs/frame_errors.csv"
RECON_DIR  = "outputs/reconstructed_frames"
HEAT_DIR   = "outputs/heatmaps"
os.makedirs(RECON_DIR, exist_ok=True)
os.makedirs(HEAT_DIR,  exist_ok=True)

# -------- Dataset --------
class FrameDS(Dataset):
    def __init__(self, root):
        self.paths = []
        for clip in sorted(os.listdir(root)):
            sub = os.path.join(root, clip)
            if os.path.isdir(sub):
                self.paths += [os.path.join(sub, f) for f in sorted(os.listdir(sub))]
        self.t = transforms.ToTensor()
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p).convert("L").resize(IMG_SIZE)
        return self.t(img), p

# -------- Autoencoder (same as training) --------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1),    nn.Sigmoid()
        )
    def forward(self,x): return self.dec(self.enc(x))

# -------- Evaluate --------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(dev); model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

dl = DataLoader(FrameDS(DATA_DIR), batch_size=1, shuffle=False)
errs, flags = [], []

with torch.no_grad():
    for timg, path in tqdm(dl, desc="Eval"):
        timg = timg.to(dev)
        recon = model(timg)[..., :IMG_SIZE[0], :IMG_SIZE[1]]  # crop safety
        mse   = nn.functional.mse_loss(recon, timg).item()
        errs.append(mse)

        # ---- save recon & heatmap ----
        recon_np = (recon[0,0].cpu().numpy()*255).astype("uint8")
        orig_np  = (timg [0,0].cpu().numpy()*255).astype("uint8")

        name = os.path.basename(path[0])
        Image.fromarray(recon_np).save(os.path.join(RECON_DIR, name))

        # heat-map (absolute diff)
        heat = Image.fromarray(np.abs(orig_np - recon_np))
        heat.save(os.path.join(HEAT_DIR, name))

# ---- CSV ----
thr = np.mean(errs) + 3*np.std(errs)
flags = [1 if e > thr else 0 for e in errs]
pd.DataFrame(
    {"frame_idx": range(len(errs)), "mse": errs, "is_anomaly": flags}
).to_csv(CSV_PATH, index=False)

print(f"✅ frame_errors.csv saved\n✅ recon & heatmap images saved\nThreshold = {thr:.5f}")
