# ShopRank

ShopRank is a two-stage recommendation system built for online retail discovery.

## Phase 1

Phase 1 establishes the machine learning foundation, data versioning, experiment tracking and model registration.

- Data Versioning: Tracked the raw MovieLens 25M dataset and processed outputs with DVC connected to DagsHub storage.
- Data Validation: Verified interactions against duplicate records, future timestamps, missing catalog identifiers and null entries.
- Preprocessing and Temporal Split: Filtered active users and items, extracted movie release years and genres, and partitioned data chronologically into train, ranker, validation and test windows.
- Baseline Models: Built four baseline systems including popular overall, recent popular, item cooccurrence and TruncatedSVD matrix factorization. TruncatedSVD is used as the matrix factorization baseline because implicit is not installed on this machine.
- Two-Stage Recommender: Trained a PyTorch two-tower retrieval model with in-batch negatives, built a FAISS IndexFlatIP candidate index, and trained a LightGBM LambdaRank re-ranking model.
- Experiment Tracking: Recorded parameters, ranking metrics and run tags on the DagsHub MLflow tracking server.
- Model Registry: Bundled neural embeddings, FAISS index and ranker booster into an MLflow pyfunc artifact registered with version tags.
- Cloud Execution: Created an automated Google Colab notebook to pull dataset files, reproduce the full DVC pipeline and push updated outputs.

This project implements a standard model pipeline and does not use RAG or agentic workflows.

## Repository Layout

- `src/`: Pipeline scripts for validation, preprocessing, feature extraction, baselines, retrieval, ranking, evaluation and model registration
- `tests/`: Unit test suite covering metrics and temporal split integrity
- `notebooks/`: Google Colab training notebook
- `data/`: DVC tracked raw and processed datasets
- `models/`: Trained model weights, FAISS index and item embeddings
- `metrics/`: Pipeline evaluation outputs and validation metrics
- `dvc.yaml`: DVC pipeline definition linking all stages

## How To Reproduce

```bash
git clone https://github.com/shayanO-O/shoprank.git
cd shoprank
pip install -r requirements.txt
dvc pull
dvc repro
```

## DagsHub Links

- MLflow Tracking: https://dagshub.com/shayanO-O/shoprank.mlflow
- Model Registry: https://dagshub.com/shayanO-O/shoprank.models

## Limitations

MovieLens contains explicit star ratings rather than real implicit user clicks. Cold start handling covers new users with recent popular recommendations, while new items are not handled yet.
