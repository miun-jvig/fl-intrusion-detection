import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def even_split(df, num_clients=5):
    # Distribute remaining datasets across clients
    df_splits = [df.iloc[i::num_clients] for i in range(num_clients)]

    for i, df_split in enumerate(df_splits):
        df_split.to_csv(PROJECT_ROOT / 'datasets' / f'preprocessed_{i}.csv', index=False, encoding='utf-8')
        print(f"Saved preprocessed datasets for device {i}")

    return df_splits


def dirichlet_split(df, num_clients=5, alpha=10.0):
    """
    Returns a dict client_id -> list of row‐indices.
    A larger alpha → *more* balanced; smaller alpha → more skew.
    """
    y = df["Attack_type"].values
    classes = np.unique(y)
    client_idxs = {i: [] for i in range(num_clients)}

    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        # sample a Dirichlet distribution over clients
        proportions = np.random.dirichlet([alpha] * num_clients)
        # convert to cumulative counts
        counts = (proportions * len(idx_c)).astype(int)
        # fix rounding
        counts[-1] = len(idx_c) - counts[:-1].sum()
        start = 0
        for i, cnt in enumerate(counts):
            client_idxs[i].extend(idx_c[start: start + cnt].tolist())
            start += cnt

    return client_idxs
