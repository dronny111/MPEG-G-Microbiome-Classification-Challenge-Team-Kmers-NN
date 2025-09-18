import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_FEAT,
    TEST_FEAT,
)


train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)


train_features = pd.read_csv(TRAIN_FEAT)
test_features  = pd.read_csv(TEST_FEAT)


train_df["filename"] = train_df["filename"].str.replace(".mgb", ".fastq")
test_df["filename"]  = test_df["filename"].str.replace(".mgb", ".fastq")


fn2cls = dict(zip(train_df.filename, train_df.SampleType))

train_features["SampleType"] = train_features["file"].map(fn2cls)
train_features["SampleType"] = train_features["SampleType"].map(
    {"Mouth": 0, "Nasal": 1, "Skin": 2, "Stool": 3}
)


_cont_features = train_features.columns.difference(
    ["ID", "file", "fold", "SampleType"]
).tolist()


scaler = StandardScaler()
train_features[_cont_features] = scaler.fit_transform(
    train_features[_cont_features]
)
test_features[_cont_features] = scaler.transform(
    test_features[_cont_features]
)

cont_features = _cont_features
train_features = train_features
test_features  = test_features
