import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar os modelos locais
model1 = joblib.load("rf_hospital_1.joblib")
model2 = joblib.load("rf_hospital_2.joblib")
models = [model1, model2]

# 2. Criar o modelo global
global_model = RandomForestClassifier()

# 3. Concatenar os estimadores (as árvores)
global_model.estimators_ = []
for m in models:
    global_model.estimators_.extend(m.estimators_)

# 4. CONFIGURAÇÃO MANUAL DOS ATRIBUTOS (O que faltava)
global_model.n_estimators = len(global_model.estimators_)
global_model.classes_ = model1.classes_
global_model.n_classes_ = len(model1.classes_)
global_model.n_features_in_ = model1.n_features_in_
global_model.n_outputs_ = 1  # Resolve o erro 'n_outputs_'

# Importante para evitar o erro de "feature names"
if hasattr(model1, "feature_names_in_"):
    global_model.feature_names_in_ = model1.feature_names_in_

# 5. Salvar o modelo final
joblib.dump(global_model, "rf_global_model.joblib")
print("[Aggregate] Modelo federado agregado e salvo com sucesso!")