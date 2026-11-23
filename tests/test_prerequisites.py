import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.two_tower import ItemTower
from src.evaluate import apply_diversity_rule
from src.generate_availability import generate_availability
from src.register import calculate_file_sha256, ShopRankWrapper

def test_item_tower_metadata_only_cold_start():
    num_items = 100
    emb_dim = 16
    num_genres = 10
    
    tower = ItemTower(num_items=num_items, emb_dim=emb_dim, num_genres=num_genres, id_dropout=0.1)
    tower.eval()
    
    batch_size = 5
    genre_vec = torch.randn(batch_size, num_genres)
    genre_vec = torch.clamp(genre_vec, min=0.0) # non-negative
    year_val = torch.tensor([[1995.0], [2000.0], [2010.0], [2020.0], [2024.0]], dtype=torch.float32)
    
    # Cold start: item_idx is None
    with torch.no_grad():
        cold_emb = tower(item_idx=None, genre_vec=genre_vec, year_val=year_val)
        
    assert cold_emb.shape == (batch_size, emb_dim)
    # Check L2 norm is ~1.0
    norms = torch.norm(cold_emb, p=2, dim=1).numpy()
    np.testing.assert_allclose(norms, np.ones(batch_size), atol=1e-5)

def test_diversity_rule():
    # 5 items from genre 0, 3 items from genre 1, 2 items from genre 2
    # Cap at 2 per genre
    item_genre_dict = {
        101: np.array([1, 0, 0], dtype=np.float32),
        102: np.array([1, 0, 0], dtype=np.float32),
        103: np.array([1, 0, 0], dtype=np.float32),
        104: np.array([1, 0, 0], dtype=np.float32),
        105: np.array([0, 1, 0], dtype=np.float32),
        106: np.array([0, 1, 0], dtype=np.float32),
        107: np.array([0, 0, 1], dtype=np.float32),
        108: np.array([0, 0, 1], dtype=np.float32),
    }
    ranked = [101, 102, 103, 104, 105, 106, 107, 108]
    diversified = apply_diversity_rule(ranked, item_genre_dict, max_per_genre=2, top_k=6)
    
    # Should contain at most 2 of genre 0: [101, 102]
    # genre 1: [105, 106]
    # genre 2: [107, 108]
    assert len(diversified) == 6
    g0_count = sum(1 for m in diversified if np.argmax(item_genre_dict[m]) == 0)
    assert g0_count == 2
    assert diversified[:4] == [101, 102, 105, 106]

def test_generate_availability(tmp_path):
    out_file = tmp_path / "availability.parquet"
    df = generate_availability(output_path=str(out_file), unavailable_ratio=0.05, seed=42)
    
    assert out_file.exists()
    assert "movieId" in df.columns
    assert "is_available" in df.columns
    assert df["is_available"].dtype == bool
    assert not df["is_available"].all() # some are False

def test_file_sha256_and_manifest(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("shoprank-artifact-content-v1")
    
    hash1 = calculate_file_sha256(f1)
    assert isinstance(hash1, str)
    assert len(hash1) == 64
    
    # Corrupting content should change hash
    f1.write_text("shoprank-artifact-corrupted")
    hash2 = calculate_file_sha256(f1)
    assert hash1 != hash2
