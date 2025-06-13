import os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATA_DIR   = "data/processed/UCSDped1/Train"
IMG_SIZE   = (227, 227)
EPOCHS     = 10
BATCH      = 16
LR         = 1e-3
MODEL_PATH = "models/autoencoder.pth"

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
        img = Image.open(self.paths[idx]).convert("L")
        img = img.resize(IMG_SIZE)
        return self.t(img)

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

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FrameDS(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = AE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    for epoch in range(EPOCHS):
        tot = 0
        for imgs in tqdm(dl, desc=f"E{epoch+1}/{EPOCHS}"):
            imgs = imgs.to(dev)
            out  = model(imgs)[..., :IMG_SIZE[0], :IMG_SIZE[1]]
            loss = crit(out, imgs)

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {epoch+1}: {tot/len(dl):.5f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🗄️ Model saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()
