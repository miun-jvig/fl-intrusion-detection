import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def preprocess_data(filename, test_size=0.05, num_clients=5):
    df = pd.read_csv(filename, low_memory=False)
    drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
                    "tcp.dstport", "udp.port", "mqtt.msg"]
    df.drop(drop_columns, axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(subset=None, keep="first", inplace=True)
    df = shuffle(df)
    df.isna().sum()
    print(df['Attack_type'].value_counts())

    # normalize numerical data
    df_feats = df.drop("Attack_type", axis=1)
    scaler = StandardScaler().fit(df_feats.select_dtypes("number"))
    df_feats[df_feats.select_dtypes("number").columns] = scaler.transform(df_feats.select_dtypes("number"))
    df = pd.concat([df_feats, df["Attack_type"]], axis=1)

    # encode
    columns_to_encode = [
        'http.request.method', 'http.referer', 'http.request.version',
        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
    ]
    for column in columns_to_encode:
        df = encode_text_dummy(df, column)

    # global test set
    remaining_df, global_test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['Attack_type']
    )
    global_test_df.to_csv(PROJECT_ROOT / 'datasets' / 'global_test.csv')
    print(f"Saved global test set with {len(global_test_df)} samples.")

    # Distribute remaining datasets across clients
    df_splits = [remaining_df.iloc[i::num_clients] for i in range(num_clients)]

    for i, df_split in enumerate(df_splits):
        df_split.to_csv(PROJECT_ROOT / 'datasets' / f'preprocessed_{i}.csv', encoding='utf-8')
        print(f"Saved preprocessed datasets for device {i}")


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    return df


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(PROJECT_ROOT / 'datasets' / 'DNN-EdgeIIoT-dataset.csv', args.test_size, args.num_clients)
