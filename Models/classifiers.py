import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from Util.preprocessor import build_preprocessor
from Models.thresholding import tune_threshold

import matplotlib.pyplot as plt
from sklearn.base import clone 


def predict_with_threshold(model, X, threshold: float = 0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def evaluate_model(model, X_test, y_test, threshold: float, title: str, file_path: str, cv=None):
    y_pred, y_proba = predict_with_threshold(model, X_test, threshold=threshold)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    
    for train_idx, val_idx in cv.split(X_test, y_test):
        X_tr, X_va = X_test.iloc[train_idx], X_test.iloc[val_idx]
        y_tr, y_va = y_test.iloc[train_idx], y_test.iloc[val_idx]

        y_pred_fold, y_proba_fold = predict_with_threshold(model, X_va, threshold=threshold)

        accuracies.append(accuracy_score(y_va, y_pred_fold))
        precisions.append(precision_score(y_va, y_pred_fold))
        recalls.append(recall_score(y_va, y_pred_fold))
        f1s.append(f1_score(y_va, y_pred_fold))
        aucs.append(roc_auc_score(y_va, y_proba_fold))

    accuracy_std = np.std(accuracies)
    precision_std = np.std(precisions)
    recall_std = np.std(recalls)
    f1_std = np.std(f1s)
    roc_auc_std = np.std(aucs)

    result_text = f"{title} | Threshold={threshold} | " \
                  f"Accuracy={accuracy:.4f} ± {accuracy_std:.4f} | " \
                  f"Precision={precision:.4f} ± {precision_std:.4f} | " \
                  f"Recall={recall:.4f} ± {recall_std:.4f} | " \
                  f"F1-score={f1:.4f} ± {f1_std:.4f} | " \
                  f"ROC AUC={roc_auc:.4f} ± {roc_auc_std:.4f}\n"

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(result_text)

    print(result_text)

def _make_cv(random_state: int = 42):
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)


def _gridsearch_model(model_name: str, estimator, param_grid: dict, X_train_raw, y_train, file_path: str, random_state: int = 42,
                      scoring: str = "roc_auc", use_randomized: bool = False, n_iter: int = 16):
    preprocessor = build_preprocessor()
    ros = RandomOverSampler(random_state=random_state)

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("oversample", ros),
        ("model", estimator)
    ])

    cv = _make_cv(random_state=random_state)

    if use_randomized:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=2,
            refit=True,
            verbose=0,
            random_state=random_state
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=2,
            refit=True,
            verbose=0
        )
    print(f"\n>>> CV per {model_name} iniziato...")
    search.fit(X_train_raw, y_train)
    print(f">>> CV per {model_name} completato.")

    best_idx = search.best_index_
    mean_score = search.cv_results_["mean_test_score"][best_idx]
    std_score = search.cv_results_["std_test_score"][best_idx]

    result_text = (
        f"=== {model_name} — CV (10-fold, scoring={scoring}) ===\n"
        f"Best params: {search.best_params_}\n"
        f"CV mean: {mean_score:.6f} | CV std: {std_score:.6f}\n"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(result_text)
    return search.best_estimator_, mean_score, std_score, cv


def train_all_models(X_train_raw, y_train, X_test_raw, y_test, random_state: int = 42):
    log_file_path = "Docs/log_results.txt"
    log_est = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=random_state)
    log_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }
    log_model, log_mean, log_std, cv_log = _gridsearch_model(
        "Logistic Regression", log_est, log_grid, X_train_raw, y_train, file_path=log_file_path, random_state=random_state
    )
    evaluate_model(log_model, X_test_raw, y_test, threshold=0.5, title="Logistic Regression", file_path=log_file_path, cv=cv_log)

    dt_file_path = "Docs/dt_results.txt"
    dt_est = DecisionTreeClassifier(random_state=random_state)
    dt_grid = {
    "model__max_depth": [None, 15],
    "model__min_samples_leaf": [1, 5],
    "model__class_weight": [None, "balanced"],
    }
    dt_model, dt_mean, dt_std, cv_dt= _gridsearch_model(
        "Decision Tree", dt_est, dt_grid, X_train_raw, y_train, file_path=dt_file_path, random_state=random_state
    )
    evaluate_model(dt_model, X_test_raw, y_test, threshold=0.5, title="Decision Tree", file_path=dt_file_path, cv=cv_dt)

    rf_file_path = "Docs/rf_results.txt"
    rf_est = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 20],
    "model__max_features": ["sqrt"],
    "model__min_samples_leaf": [1, 4],
    "model__class_weight": ["balanced_subsample"],
    }
    rf_model, rf_mean, rf_std, cv_rf = _gridsearch_model(
        "Random Forest", rf_est, rf_grid, X_train_raw, y_train, file_path=rf_file_path, random_state=random_state
    )
    best_rf = tune_threshold(rf_model, X_test_raw, y_test, label="RandomForest")
    evaluate_model(rf_model, X_test_raw, y_test, threshold=best_rf["thr"], title="Random Forest (tuned threshold)", file_path=rf_file_path, cv=cv_rf)

    knn_file_path = "Docs/knn_results.txt"
    knn_est = KNeighborsClassifier()
    knn_grid = {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["minkowski"],
    }
    knn_model, knn_mean, knn_std, cv_knn = _gridsearch_model(
        "KNN", knn_est, knn_grid, X_train_raw, y_train, file_path=knn_file_path, random_state=random_state
    )
    evaluate_model(knn_model, X_test_raw, y_test, threshold=0.5, title="KNN", file_path=knn_file_path, cv=cv_knn)

    mlp_file_path = "Docs/mlp_results.txt"
    mlp_est = MLPClassifier(
        random_state=random_state,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-3
    )
    mlp_params = {
    "model__hidden_layer_sizes": [(64,), (64, 32)],
    "model__alpha": [1e-4, 1e-3],
    "model__learning_rate_init": [1e-3],
    }
    mlp_model, mlp_mean, mlp_std, cv_mlp = _gridsearch_model(
        "MLP",
        mlp_est,
        mlp_params,
        X_train_raw,
        y_train,
        file_path=mlp_file_path,
        random_state=random_state,
        use_randomized=True,
        n_iter=4
    )
    evaluate_model(mlp_model, X_test_raw, y_test, threshold=0.5, title="MLP", file_path=mlp_file_path, cv=cv_mlp)

    print("\n>>> Generazione curve ROC per ogni modello...")
    plot_mean_roc_cv(log_model, X_train_raw, y_train, "LogReg - ROC (10-fold CV)", "Docs/roc_logreg_cv.png")
    plot_mean_roc_cv(dt_model,  X_train_raw, y_train, "DT - ROC (10-fold CV)",     "Docs/roc_dt_cv.png")
    plot_mean_roc_cv(rf_model,  X_train_raw, y_train, "RF - ROC (10-fold CV)",     "Docs/roc_rf_cv.png")
    plot_mean_roc_cv(knn_model, X_train_raw, y_train, "KNN - ROC (10-fold CV)",    "Docs/roc_knn_cv.png")
    plot_mean_roc_cv(mlp_model, X_train_raw, y_train, "MLP - ROC (10-fold CV)",    "Docs/roc_mlp_cv.png")

    return log_model, dt_model, rf_model, knn_model, mlp_model


def plot_mean_roc_cv(estimator, X, y, title, out_path, n_splits=10, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    fig, ax = plt.subplots()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        est = clone(estimator)
        est.fit(X_tr, y_tr)

        y_score = est.predict_proba(X_va)[:, 1]
        fpr, tpr, _ = roc_curve(y_va, y_score)
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        ax.plot(fpr, tpr, alpha=0.12)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.plot(mean_fpr, mean_tpr, linewidth=2, label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.15, label="± 1 std (TPR)")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return mean_auc, std_auc
