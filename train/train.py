import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Dataset
# ----------------------------
class OTDRDataset(Dataset):
    def __init__(self, tensor_folder):
        self.tensor_folder = tensor_folder
        self.files = [f for f in os.listdir(tensor_folder) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.tensor_folder, self.files[idx])
        x = np.load(path)  # (3, 224, 224)
        x = torch.from_numpy(x).float()
        return x, self.files[idx]

# ----------------------------
# 2️⃣ Autoencoder
# ----------------------------
class OTDRAutoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Encoder: simple CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 3x224x224 -> 32x112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256x14x14
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 512x7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*7*7, embedding_dim)
        )
        # Decoder: mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512*7*7),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # normalize output to 0-1
        )

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return out, emb

# ----------------------------
# 3️⃣ Training setup
# ----------------------------
tensor_folder = "./output/gaf_mtf_tensors"
dataset = OTDRDataset(tensor_folder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OTDRAutoencoder(embedding_dim=128).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 30

# ----------------------------
# 4️⃣ Training loop
# ----------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for x, _ in dataloader:
        x = x.to(device)
        x = x / x.max()  # optional normalization per batch
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch}/{num_epochs}] Loss: {epoch_loss:.6f}")

# ----------------------------
# 5️⃣ Extract embeddings
# ----------------------------
model.eval()
embeddings = []
filenames = []

with torch.no_grad():
    for x, fnames in dataloader:
        x = x.to(device)
        x = x / x.max()
        _, emb = model(x)
        embeddings.append(emb.cpu())
        filenames.extend(fnames)

embeddings = torch.cat(embeddings).numpy()
print("Embeddings shape:", embeddings.shape)

# Save embeddings for downstream tasks
np.save("otdr_embeddings.npy", embeddings)
with open("otdr_filenames.txt", "w") as f:
    f.write("\n".join(filenames))

# ----------------------------
# 6️⃣ Optional visualization with t-SNE
# ----------------------------
from sklearn.manifold import TSNE
emb_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1])
plt.title("OTDR Embeddings t-SNE")
plt.show()