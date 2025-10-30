from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.config import TARGET, MODELS_DIR, RANDOM_STATE


def train(train_df, valid_df):
    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    x_valid = valid_df.drop(columns=[TARGET])
    y_valid = valid_df[TARGET]

    #ajustamos el weight para el desbalanceo
    scale_pos_weight = (len(y_train) - y_train.sum() / y_train.sum())

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
    }

    evals = [(dtrain, "train"), (dvalid, 'valid')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    save_model(model, 'xgb_model.pkl')
    return model

def evaluate_model(model, test_df):
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    dtest = xgb.DMatrix(x_test)
    y_proba = model.predict(dtest)
    auc = roc_auc_score(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    return {'roc_auc': auc, 'report': report, 'cm': cm}

def save_model(model, name:str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / name)

def load_model(name: str):
    return joblib.load(MODELS_DIR / name)

def save_artifacts(art, name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, MODELS_DIR / name)

def load_artifacts(name: str):
    return joblib.load(MODELS_DIR / name)