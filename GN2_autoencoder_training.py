import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feature_names = ["eta", "phi", "pt_NOSYS", "GN2v01_quantile_2D", "E_NOSYS"]
num_features = len(feature_names)


# ============================================
# 1. LOAD AND PREPROCESS DATA
# ============================================
with h5py.File("2b_jets.h5", "r") as f:
    jets = f["jets"]
    events = f["events"]
    subset_size = 200000
    offset = 800000
    events_subset= events[offset:offset+subset_size]
    jets_all = np.stack(
        [f["jets"][feat][offset:offset+subset_size] for feat in feature_names],
        axis=-1
    )

    valid_mask_all = f["jets"]["valid"][offset:offset+subset_size].astype(bool)
    origin_jets_all = f["jets"]["is_from_top_gluon"][offset:offset+subset_size]

max_jets_per_event = jets_all.shape[1]
num_events = jets_all.shape[0]

# Allocate padded arrays
jets_padded = np.zeros((num_events, max_jets_per_event, num_features), dtype=np.float32)
labels_padded = np.full((num_events, max_jets_per_event), -1000, dtype=np.int64)

# Apply valid mask + pad/truncate
for evt in range(num_events):
    jets_evt = jets_all[evt][valid_mask_all[evt]]
    labels_evt = origin_jets_all[evt][valid_mask_all[evt]]

    length = min(len(jets_evt), max_jets_per_event)
    jets_padded[evt, :length] = jets_evt[:length]
    labels_padded[evt, :length] = labels_evt[:length]
    # padding beyond `length` is already handled by array initialization


# Convert to torch
jets_tensor = torch.tensor(jets_padded, dtype=torch.float32)
labels_tensor = torch.tensor(labels_padded, dtype=torch.long)

jets_tensor = jets_tensor.to(device)
labels_tensor = labels_tensor.to(device)


print(f"Jets tensor shape: {jets_tensor.shape}")  # (B, N, F)
print(f"Labels shape: {labels_tensor.shape}")    # (B,N,)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(jets_tensor, labels_tensor, test_size=0.2, random_state=42)
def normalize_with_mask_torch(X_train, X_val, pad_value=-1000):
    mask_train = (X_train != pad_value)

    # Compute mean/std from non-padded entries
    mean = (X_train * mask_train).sum(dim=(0, 1)) / mask_train.sum(dim=(0, 1))
    std = torch.sqrt(((X_train - mean) * mask_train).pow(2).sum(dim=(0, 1)) /
                     mask_train.sum(dim=(0, 1)))
    std[std == 0] = 1  # avoid div-by-zero

    def _normalize(X, mean, std):
        mask = (X != pad_value)
        X_norm = X.clone()
        # Broadcast mean/std over (B, N, F)
        X_norm[mask] = ((X - mean.unsqueeze(0).unsqueeze(0)) /
                        std.unsqueeze(0).unsqueeze(0))[mask]
        return X_norm

    return _normalize(X_train, mean, std), _normalize(X_val, mean, std), mean, std

X_train, X_val, mean, std = normalize_with_mask_torch(X_train, X_val, pad_value=-1000)


# ============================================
# 2. AUTOENCODER PRETRAINING
# ============================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

embed_dim = 256
autoencoder = Autoencoder(input_dim=len(feature_names), embed_dim=embed_dim).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

BATCH_SIZE = 1000
EPOCHS = 50
patience = 20  # stop if no improvement for 20 epochs
best_val_loss = float("inf")
wait = 0

X_train = X_train.to(device)
X_val   = X_val.to(device)

# --- training ---
autoencoder.train()
best_val_loss = float('inf')
wait = 0
patience = 20

for epoch in range(EPOCHS):
    # TRAIN
    perm = torch.randperm(X_train.size(0), device=device)
    epoch_train_loss = 0.0
    for i in range(0, X_train.size(0), BATCH_SIZE):
        batch = X_train[perm[i:i+BATCH_SIZE]]
        optimizer.zero_grad()
        recon = autoencoder(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch.size(0)
    epoch_train_loss /= X_train.size(0)

    # VAL
    autoencoder.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for i in range(0, X_val.size(0), BATCH_SIZE):
            batch = X_val[i:i+BATCH_SIZE]
            recon = autoencoder(batch)
            vloss = criterion(recon, batch)
            epoch_val_loss += vloss.item() * batch.size(0)
    autoencoder.train()
    epoch_val_loss /= X_val.size(0)

    print(f"[Autoencoder] Epoch {epoch+1}/{EPOCHS} - Train loss: {epoch_train_loss:.4f} - Val loss: {epoch_val_loss:.4f}")

    # early stopping + best weights
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        wait = 0
        best_encoder_weights = {k: v.detach().cpu().clone() for k, v in autoencoder.encoder.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# use the best encoder weights (not last)
pretrained_encoder_weights = best_encoder_weights

for k, v in pretrained_encoder_weights.items():
    print(k, v.shape)

# 3. DEFINE GN2-LIKE ARCHITECTURE
# ============================================
class Initializer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
    def forward(self, x):
        return self.encoder(x)

class GlobalAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

class GN2(nn.Module):
    def __init__(self, jets_dim, embed_dim, num_classes):
        super().__init__()
        self.encoder = Initializer(jets_dim, embed_dim)
        self.transformers = TransformerEncoder(embed_dim)
        self.pooling = GlobalAttentionPooling(embed_dim)
        self.classifier = Classifier(embed_dim * 2, num_classes)  # now takes jet + pooled

    def forward(self, jets):
        # Encode jets
        x = self.encoder(jets)              # (B, N, D)
        x = self.transformers(x)             # (B, N, D)

        # Global pooled representation
        pooled = self.pooling(x)              # (B, D)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, D)

        # Concatenate local jet + global context
        x_cat = torch.cat([x, pooled_expanded], dim=-1)  # (B, N, 2D)

        # Classify each jet individually
        return self.classifier(x_cat)  # (B, N, num_classes)


from torch.utils.data import TensorDataset, DataLoader

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=1000, lr=1e-4, ignore_index=-1000,
                max_grad_norm=1.0):

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        total_valid = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

        # NaN/Inf checks
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                raise ValueError("NaN/Inf in X_batch")
            if torch.isnan(y_batch).any() or torch.isinf(y_batch).any():
                raise ValueError("NaN/Inf in y_batch")

            optimizer.zero_grad()
            outputs = model(X_batch)  # shape [batch, N, num_classes]

            outputs_flat = outputs.view(-1, outputs.size(-1))
            y_flat = y_batch.view(-1)

            if torch.isnan(outputs_flat).any() or torch.isinf(outputs_flat).any():
                raise ValueError("NaN/Inf in outputs")

            loss = criterion(outputs_flat, y_flat)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN/Inf in loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Accumulate weighted loss
            num_valid = (y_flat != ignore_index).sum().item()
            epoch_train_loss += loss.item() * num_valid
            total_valid += num_valid

        # Average over valid elements
        epoch_train_loss /= total_valid
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_valid = 0

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                outputs_val = model(X_val_batch)
                outputs_val_flat = outputs_val.view(-1, outputs_val.size(-1))
                y_val_flat = y_val_batch.view(-1)

                loss_val = criterion(outputs_val_flat, y_val_flat)

                num_valid_val = (y_val_flat != ignore_index).sum().item()
                val_loss_total += loss_val.item() * num_valid_val
                val_valid += num_valid_val

        val_loss_avg = val_loss_total / val_valid
        val_losses.append(val_loss_avg)

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_train_loss:.4f}, Val loss: {val_loss_avg:.4f}")

    return train_losses, val_losses

# 5. TRAIN BOTH VERSIONS
# ============================================
# GN2 from scratch
model_scratch = GN2(jets_dim=len(feature_names), embed_dim=embed_dim, num_classes=3).to(device)
loss_scratch_train, loss_scratch_val = train_model(model_scratch, X_train, y_train, X_val, y_val, epochs=50)

# GN2 with pretrained encoder
model_pretrained = GN2(jets_dim=len(feature_names), embed_dim=embed_dim, num_classes=3).to(device)

model_pretrained.encoder.net.load_state_dict(pretrained_encoder_weights, strict=False)
loss_pre_train, loss_pre_val = train_model(model_pretrained, X_train, y_train, X_val, y_val, epochs=50)

# ============================================
# 6. PLOT RESULTS
# ============================================
plt.figure(figsize=(8,5))
#plt.plot(loss_scratch_train, label="Scratch - Train")
plt.plot(loss_scratch_val, label="Scratch - Val")
#plt.plot(loss_pre_train, label="Pretrained - Train")
plt.plot(loss_pre_val, label="Pretrained - Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("GN2 Training: Scratch vs Pretrained Encoder")
plt.savefig("/content/drive/MyDrive/gn2_train_pretrain_val_loss_200000_50_5.pdf")
plt.show()