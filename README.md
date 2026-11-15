# ShopRank

ShopRank recommends items to users, in this case movies. A recommender system decides what a person sees first on a page, so ranking quality matters much more than plain accuracy.

We built this recommender in two stages. Stage one quickly finds a few hundred likely items from thousands of options. Stage two re-orders those items with a smarter model and returns the best ten.

## Remote Links

| Name | Link |
|---|---|
| DagsHub Repository | https://dagshub.com/shayanO-O/shoprank |

## Repository Layout

- `src/`: pipeline scripts for data preparation, models, evaluation and registration
- `tests/`: unit tests for evaluation metrics and time splitting
- `notebooks/`: the Google Colab notebook used for cloud training
- `data/`: raw and processed data folders
- `models/`: saved model weights, FAISS index and item vectors
- `metrics/`: validation results, baseline scores and final evaluation files
- `dvc.yaml`: pipeline definition that links all stages together
- `params.yaml`: shared settings and hyperparameter values

Files in `data/` and `models/` are tracked by DVC and stored on DagsHub, so they are not stored in GitHub.

## Phase 1

Phase 1 covers the core machine learning pipeline, data versioning, experiment tracking and model registration.

- We track the raw MovieLens dataset and all processed parquet files with DVC using DagsHub cloud storage.
- We run data quality checks for duplicate rows, future dates and missing values, then split the data by time into train, ranker, validation and test sets.
- We evaluate four baselines: popular overall, recent popular, item cooccurrence and matrix factorization. We used TruncatedSVD for matrix factorization because the implicit package would not install on our machine.
- We built a two-stage recommender using a PyTorch two-tower neural network with a FAISS index for candidate retrieval and a LightGBM LambdaRank model for re-ranking.
- We track all parameters, metrics and artifacts with MLflow on DagsHub, and we register the champion model version.
- We built a Google Colab notebook that runs the full pipeline from data pull to model registration.

This project is a standard model pipeline. We do not use RAG or agentic workflows.

### Results

These are the test results, and we compare our main two-stage system with the four baselines.

| Model | Val NDCG@10 | Val Recall@50 | Test NDCG@10 | Test Recall@50 | Catalog Coverage | Diversity |
|---|---|---|---|---|---|---|
| Popular Overall | 0.2735 | 0.1623 | - | - | 0.0156 | 0.7277 |
| Recent Popular | 0.3070 | 0.1997 | - | - | 0.0100 | 0.6733 |
| Item Cooccurrence | 0.3052 | 0.1969 | - | - | 0.0293 | 0.6746 |
| Matrix Factorization | 0.3077 | 0.1985 | - | - | 0.0619 | 0.6713 |
| Retrieval Only | 0.2656 | 0.1563 | 0.2544 | 0.1421 | 0.1879 | 0.7036 |
| Retrieval + Ranker | 0.3299 | 0.1873 | 0.3203 | 0.1751 | 0.1051 | 0.6523 |
| Cold Start Only | 0.3757 | 0.2083 | 0.3564 | 0.1927 | 0.0010 | 0.6662 |

NDCG@10 measures ranking quality in the top ten results, and Recall@50 measures the share of relevant items found in the top fifty candidates.

## How To Reproduce

Follow these steps to run the project:

1. Clone the repo and enter the folder.
```bash
git clone https://github.com/shayanO-O/shoprank.git
cd shoprank
```

2. Install the packages with pip.
```bash
pip install -r requirements.txt
```

3. Add your DagsHub username and token to a `.env` file. You can check `.env.example` to see the format.
```bash
cp .env.example .env
```

4. Pull the dataset files from DagsHub.
```bash
dvc pull
```

5. Run the full pipeline.
```bash
dvc repro
```

The full training run is meant for Google Colab with a GPU. You can run `notebooks/train_colab.ipynb` directly.

## Limitations

MovieLens contains star ratings instead of real user clicks. New items are not handled yet, and new users receive the most popular recent items.
