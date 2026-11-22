"""Train and evaluate a deposit subscription classifier."""

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.get_data import DATA_PATH, ROOT

ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit("Data missing. Run: uv run python -m src.get_data")

    data = pd.read_csv(DATA_PATH, sep=";")
    # Call duration is unavailable when making a prediction before the call.
    features = data.drop(columns=["y", "duration"])
    target = data["y"].map({"no": 0, "yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), make_column_selector(dtype_include="number")),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    model = make_pipeline(preprocessing, LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "random_state": RANDOM_STATE,
        "positive_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, model.predict(x_test))),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    metrics_json = json.dumps(metrics, indent=2, allow_nan=False)
    with (ARTIFACTS / "model.pkl").open("wb") as output:
        pickle.dump(model, output, protocol=pickle.HIGHEST_PROTOCOL)
    (ARTIFACTS / "metrics.json").write_text(metrics_json + "\n", encoding="utf-8")
    print("Saved artifacts/model.pkl and artifacts/metrics.json")
    print(metrics_json)


if __name__ == "__main__":
    main()
