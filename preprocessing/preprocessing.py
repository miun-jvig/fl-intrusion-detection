import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def load_data(filename):
    df = pd.read_csv(filename, low_memory=False)
    df.head(5)
    labels = 'Attack_type'
    print(df[labels].value_counts())

    x = df.drop(columns=[labels]).to_numpy().astype('float32')  # Features: all columns except 'Attack_type'
    y = to_categorical(LabelEncoder().fit_transform(df[labels]))  # Label: 'Attack_type', one hot encoded
    return x, y


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

    # encode
    columns_to_encode = [
        'http.request.method', 'http.referer', 'http.request.version',
        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
    ]
    for column in columns_to_encode:
        df = encode_text_dummy(df, column)

    # save as new .csv-file
    df.to_csv('preprocessed_DNN.csv', encoding='utf-8')


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    return df
