import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def tune_threshold(model, X, y, label: str = "Model", thresholds=None, print_table: bool = True):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    y_proba = model.predict_proba(X)[:, 1]

    if print_table:
        print(f"\n=== Tuning soglia — {label} ===")
        print("thr   prec_1  rec_1   f1_1   FP   FN")
        print("----------------------------------------")

    best = {
        "thr": 0.5,
        "prec_1": -1.0,
        "rec_1": -1.0,
        "f1_1": -1.0,
        "fp": None,
        "fn": None,
    }

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average=None, labels=[0, 1], zero_division=0)
        prec_1, rec_1, f1_1 = float(prec[1]), float(rec[1]), float(f1[1])

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])

        if print_table:
            print(f"{thr:0.2f}  {prec_1:0.4f}  {rec_1:0.4f}  {f1_1:0.4f}  {fp:4d}  {fn:3d}")

        if f1_1 > best["f1_1"]:
            best = {
                "thr": round(float(thr), 2),
                "prec_1": prec_1,
                "rec_1": rec_1,
                "f1_1": f1_1,
                "fp": fp,
                "fn": fn
            }

    print(
        f"\n>>> Miglior soglia (max F1 classe 1): {best['thr']:.2f} "
        f"| prec_1={best['prec_1']:.4f} rec_1={best['rec_1']:.4f} f1_1={best['f1_1']:.4f} "
        f"| FP={best['fp']} FN={best['fn']}"
    )

    return best
