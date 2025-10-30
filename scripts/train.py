from src.data_utils import load_data, split_data, fit_preprocess, apply_preprocess
from src.model_utils import (train, evaluate_model, save_artifacts)
from src.config import TARGET

def main():
    df = load_data()
    train_df, valid_df, test_df = split_data(df)

    artifacts = fit_preprocess(train_df)
    train_df = apply_preprocess(train_df, artifacts)
    valid_df = apply_preprocess(valid_df, artifacts)
    test_df = apply_preprocess(test_df, artifacts)

    save_artifacts(artifacts, "preprocess.pkl")

    model = train(train_df, valid_df)

    metrics = evaluate_model(model, test_df)
    print(f'ROC_AUC:  {round(metrics["roc_auc"], 2)}')
    print(f'Report: {metrics["report"]}')
    print(f'Confusion Matrix: \n {metrics["cm"]}')

if __name__ == "__main__":
    main()
