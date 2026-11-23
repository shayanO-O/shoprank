import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, setup_mlflow, get_paths
from src.features import build_genre_matrix, extract_user_histories

# PyTorch Dataset For Interaction Sequences
class RetrievalDataset(Dataset):
    def __init__(self, user_histories, interactions, item_to_idx, max_len=50):
        self.samples = []
        self.item_to_idx = item_to_idx
        self.max_len = max_len
        
        positives = interactions[interactions["label"] == 1].sort_values("timestamp")
        for _, row in positives.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            if mid in item_to_idx:
                hist = user_histories.get(uid, [])
                hist_indices = [item_to_idx[m] for m in hist if m in item_to_idx and m != mid][-max_len:]
                self.samples.append((hist_indices, item_to_idx[mid]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        hist_len = min(len(hist), self.max_len)
        padded_hist = np.zeros(self.max_len, dtype=np.int64)
        if hist_len > 0:
            padded_hist[:hist_len] = hist[:hist_len]
        return torch.tensor(padded_hist, dtype=torch.long), torch.tensor(hist_len, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# Item Tower Neural Network
class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres, emb_dim, id_dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.genre_proj = nn.Linear(num_genres, emb_dim)
        self.year_proj = nn.Linear(1, emb_dim)
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.id_dropout = id_dropout
        
    def forward(self, item_idx=None, genre_vec=None, year_val=None):
        g_vec = self.genre_proj(genre_vec)
        y_vec = self.year_proj(year_val)
        if item_idx is not None:
            id_vec = self.item_emb(item_idx)
            if self.training and self.id_dropout > 0.0:
                mask = (torch.rand(id_vec.shape[0], 1, device=id_vec.device) > self.id_dropout).float()
                id_vec = id_vec * mask
            out = self.fc(id_vec + g_vec + y_vec)
        else:
            # Metadata-only path for new / cold-start items
            out = self.fc(g_vec + y_vec)
        return F.normalize(out, p=2, dim=-1)

# User Tower Neural Network
class UserTower(nn.Module):
    def __init__(self, num_items, emb_dim, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.len_proj = nn.Linear(1, emb_dim)
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.max_len = max_len
        
    def forward(self, hist_indices, hist_len):
        embs = self.item_emb(hist_indices)
        denom = torch.clamp(hist_len.unsqueeze(-1), min=1.0)
        hist_mean = torch.sum(embs, dim=1) / denom
        norm_len = (hist_len.unsqueeze(-1) / float(self.max_len))
        len_vec = self.len_proj(norm_len)
        out = self.fc(hist_mean + len_vec)
        return F.normalize(out, p=2, dim=-1)

# Full Two Tower Recommendation Architecture
class TwoTowerModel(nn.Module):
    def __init__(self, num_items, num_genres, emb_dim):
        super().__init__()
        self.user_tower = UserTower(num_items, emb_dim)
        self.item_tower = ItemTower(num_items, num_genres, emb_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, hist_indices, hist_len, item_idx, genre_vec, year_val):
        u_emb = self.user_tower(hist_indices, hist_len)
        i_emb = self.item_tower(item_idx, genre_vec, year_val)
        return u_emb, i_emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    paths = get_paths(params, smoke=args.smoke)
    paths["models"].mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_parquet(paths["processed"] / "train.parquet")
    items_df = pd.read_parquet(paths["processed"] / "items.parquet")
    
    all_items = items_df["movieId"].tolist()
    item_to_idx = {mid: i for i, mid in enumerate(all_items)}
    num_items = len(all_items)
    
    genre_matrix, genre_map = build_genre_matrix(items_df)
    years = ((items_df["year"].to_numpy().astype(np.float32) - 1900.0) / 100.0).reshape(-1, 1)
    
    emb_dim = params["retrieval"]["smoke_embedding_dim"] if args.smoke else params["retrieval"]["embedding_dim"]
    epochs = params["retrieval"]["smoke_epochs"] if args.smoke else params["retrieval"]["epochs"]
    batch_size = 64 if args.smoke else params["retrieval"]["batch_size"]
    
    user_hist = extract_user_histories(train_df, max_len=params["retrieval"]["max_history_len"])
    dataset = RetrievalDataset(user_hist, train_df, item_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = TwoTowerModel(num_items, len(genre_map), emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["retrieval"]["lr"])
    
    # Train Model With In Batch Negatives
    model.train()
    genre_tensor = torch.tensor(genre_matrix, dtype=torch.float32)
    year_tensor = torch.tensor(years, dtype=torch.float32)
    
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for hist_idx, hist_len, target_idx in loader:
            optimizer.zero_grad()
            batch_genres = genre_tensor[target_idx]
            batch_years = year_tensor[target_idx]
            u_emb, i_emb = model(hist_idx, hist_len, target_idx, batch_genres, batch_years)
            logits = torch.matmul(u_emb, i_emb.T) / 0.07
            labels = torch.arange(len(u_emb), dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            if args.smoke and count >= 50:
                break
        print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / max(1, count):.4f}")

    # Compute And Export Normalized Item Vectors
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(num_items, dtype=torch.long)
        item_vectors = model.item_tower(all_indices, genre_tensor, year_tensor).cpu().numpy()
        
    torch.save(model.state_dict(), paths["models"] / "two_tower.pt")
    np.save(paths["models"] / "item_embeddings.npy", item_vectors)
    with open(paths["models"] / "item_id_map.json", "w", encoding="utf-8") as f:
        json.dump(all_items, f)
    print("Two tower model trained and item embeddings saved successfully")
