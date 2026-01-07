import pandas as pd
import joblib

model = joblib.load("rf_global_model.joblib")

features = ["bpm", "rolling_mean", "rolling_std", "diff"]

print("\n--- Teste de Inferência (Modelo Federado) ---")

novo_dado = pd.DataFrame(
    [[120, 115, 10.5, 5]],
    columns=features
)

predicao = model.predict(novo_dado)
probabilidade = model.predict_proba(novo_dado)

print(f"Dados de entrada: {novo_dado.values}")
print(
    f"Classificação: "
    f"{'Atividade/Anormal' if predicao[0] == 1 else 'Repouso'}"
)
print(
    f"Certeza do modelo: "
    f"{probabilidade[0][int(predicao[0])]:.2f}"
)