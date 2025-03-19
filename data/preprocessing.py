import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

num_clients = 5


def preprocess_data(filename, test_size=0.05):
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
    global_test_df.to_csv("../datasets/global_test.csv", index=False)
    print(f"Saved global test set with {len(global_test_df)} samples.")

    # Distribute remaining datasets across clients
    df_splits = [remaining_df.iloc[i::num_clients] for i in range(num_clients)]

    for i, df_split in enumerate(df_splits):
        df_split.to_csv(f'../datasets/preprocessed_{i}.csv', index=False, encoding='utf-8')
        print(f"Saved preprocessed datasets for device {i}")


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    return df


preprocess_data('../datasets/DNN-EdgeIIoT-dataset.csv')
