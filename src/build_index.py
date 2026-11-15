import argparse
import sys
from pathlib import Path
import numpy as np
import faiss

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, get_paths

# Build FAISS Inner Product Index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    paths = get_paths(params, smoke=args.smoke)
    
    # Load Item Embeddings
    emb_path = paths["models"] / "item_embeddings.npy"
    embeddings = np.load(emb_path)
    
    # Construct And Save FAISS Index
    index = build_faiss_index(embeddings)
    index_out = paths["models"] / "faiss.index"
    faiss.write_index(index, str(index_out))
    print(f"FAISS index built and saved with {index.ntotal} vectors")
