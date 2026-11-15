import pandas as pd
import numpy as np

# Test Non Overlapping Temporal Windows
def test_temporal_split_order():
    timestamps = np.arange(100)
    df = pd.DataFrame({
        "userId": np.repeat(np.arange(10), 10),
        "movieId": np.tile(np.arange(10), 10),
        "rating": np.ones(100) * 4.0,
        "timestamp": timestamps
    })
    
    t_train = df["timestamp"].quantile(0.70)
    t_ranker = df["timestamp"].quantile(0.80)
    t_val = df["timestamp"].quantile(0.90)
    
    train = df[df["timestamp"] <= t_train]
    ranker = df[(df["timestamp"] > t_train) & (df["timestamp"] <= t_ranker)]
    val = df[(df["timestamp"] > t_ranker) & (df["timestamp"] <= t_val)]
    test = df[df["timestamp"] > t_val]
    
    assert train["timestamp"].max() <= ranker["timestamp"].min()
    assert ranker["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()

# Test Cold User Flagging Logic
def test_cold_user_flag():
    train_users = {1, 2, 3}
    eval_users = [1, 2, 4, 5]
    is_cold = [int(u not in train_users) for u in eval_users]
    assert is_cold == [0, 0, 1, 1]
