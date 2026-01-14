from Util.preprocessor import build_clean_dataset_from_raw, create_reduced_dataset, load_reduced_dataset, prepare_train_test_raw
from Models.classifiers import train_all_models
from KnowledgeBase.kb import KnowledgeBase

print(">>> Costruzione dataset CLEAN da RAW...")
build_clean_dataset_from_raw()

print("\n>>> Creazione dataset RIDOTTO...")
create_reduced_dataset(n_nonfraud=25000, random_state=42)

print("\n>>> Caricamento dataset RIDOTTO per verifica...")
df_reduced = load_reduced_dataset()
print("Shape dataset ridotto:", df_reduced.shape)
print("\nDistribuzione is_fraud nel dataset ridotto:")
print(df_reduced["is_fraud"].value_counts().to_string())
print("\nDistribuzione is_fraud (percentuali):")
print(df_reduced["is_fraud"].value_counts(normalize=True).to_string())

X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_raw(
    test_size=0.2,
    random_state=42
)

log_model, dt_model, rf_model, knn_model, mlp_model = train_all_models(
    X_train_raw=X_train_raw,
    y_train=y_train,
    X_test_raw=X_test_raw,
    y_test=y_test,
    random_state=42,
)

print("\n>>> Preparazione transazioni di Test con Prolog...")
probas = rf_model.predict_proba(X_test_raw)[:, 1]

kb = KnowledgeBase("KnowledgeBase/main.pl")

print("\n=== Prolog (10 transazioni di Test) ===")
for i in range(10):
    row = X_test_raw.iloc[i].to_dict()
    p = float(probas[i])

    risk_level, score_raw, score_norm, reasons, action = kb.is_risky(row, p)

    reasons_str = ", ".join(reasons) if reasons else "(none)"
    print(
        f"\n[{i}] ML proba = {p:.4f} | Prolog = {risk_level} | "
        f"ScoreRaw = {score_raw} | Score = {score_norm}/100 | Action = {action}"
    )
    print(f"  Reasons: {reasons_str}")