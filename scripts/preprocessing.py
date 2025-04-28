import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from data_loading.data_splitter import dirichlet_split, even_split
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def preprocess_data(filename, test_size=0.05):
    df_data = pd.read_csv(filename, low_memory=False)
    drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
                    "tcp.dstport", "udp.port", "mqtt.msg"]
    df_data.drop(drop_columns, axis=1, inplace=True)
    df_data.dropna(axis=0, how='any', inplace=True)
    df_data.drop_duplicates(subset=None, keep="first", inplace=True)
    df_data = shuffle(df_data)
    df_data.isna().sum()
    print(df_data['Attack_type'].value_counts())

    # normalize numerical data
    df_feats = df_data.drop("Attack_type", axis=1)
    scaler = StandardScaler().fit(df_feats.select_dtypes("number"))
    df_feats[df_feats.select_dtypes("number").columns] = scaler.transform(df_feats.select_dtypes("number"))
    df_data = pd.concat([df_feats, df_data["Attack_type"]], axis=1)

    # encode
    columns_to_encode = [
        'http.request.method', 'http.referer', 'http.request.version',
        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
    ]
    for column in columns_to_encode:
        df_data = encode_text_dummy(df_data, column)

    # global test set
    remaining_df, global_test_df = train_test_split(
        df_data, test_size=test_size, random_state=42, stratify=df_data['Attack_type']
    )
    # global_test_df.to_csv(PROJECT_ROOT / 'datasets' / 'global_test.csv', index=False, encoding='utf-8')
    print(f"Saved global test set with {len(global_test_df)} samples.")

    return remaining_df


def encode_text_dummy(df_data, name):
    dummies = pd.get_dummies(df_data[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df_data[dummy_name] = dummies[x]
    df_data.drop(name, axis=1, inplace=True)
    return df_data


def save_client_csvs(df_client, client_idx, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, idx in client_idx.items():
        print(f"Saved dirichlet split dataset preprocessed datasets for device {i}")
        df_client.iloc[idx].to_csv(out_dir / f"preprocessed_{cid}.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data from the EdgeIIoT-set")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="The test-size of the global testing set",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of clients to be used for training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help=""
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = preprocess_data(PROJECT_ROOT / "datasets" / "DNN-EdgeIIoT-dataset.csv", args.test_size)

    client_idxs = dirichlet_split(df, args.num_clients, args.alpha)
    save_client_csvs(df, client_idxs, PROJECT_ROOT / "datasets")
