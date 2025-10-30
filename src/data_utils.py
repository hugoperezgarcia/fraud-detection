import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (DATA_PATH, TARGET, SCALE_COLS, TEST_SIZE, VAL_SIZE, RANDOM_STATE)

def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        stratify=train_df[TARGET],
        random_state=RANDOM_STATE
    )

    return train_df, valid_df, test_df

def fit_preprocess(train_df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(train_df[SCALE_COLS])
    return {'scaler': scaler}

def apply_preprocess(df: pd.DataFrame, art: dict):
    df = df.copy()
    scaler = art['scaler']
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df