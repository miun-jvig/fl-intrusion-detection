import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler






def preprocess_data(filename):
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
    drop_cols = ["Attack_label", "Attack_type"]
    df_feats = df.drop(columns=[c for c in drop_cols if c in df.columns])
    scaler = StandardScaler().fit(df_feats.select_dtypes("number"))
    df_feats[df_feats.select_dtypes("number").columns] = scaler.transform(df_feats.select_dtypes("number"))
    df = pd.concat([df_feats, df[drop_cols]], axis=1)

    # encode
    columns_to_encode = [
        'http.request.method', 'http.referer', 'http.request.version',
        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
    ]
    for column in columns_to_encode:
        df = encode_text_dummy(df, column)

    # save as new .csv-file
    df.to_csv('preprocessed_DNN.csv', index=False, encoding='utf-8')


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    return df
