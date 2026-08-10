import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix

warnings.filterwarnings("ignore")

from config import *  
from data import *
from model import *

init(autoreset=True)

C_CYAN = Fore.CYAN + Style.BRIGHT
C_GREEN = Fore.GREEN + Style.BRIGHT
C_YELLOW = Fore.YELLOW + Style.BRIGHT
C_MAGENTA = Fore.MAGENTA + Style.BRIGHT
C_BLUE = Fore.BLUE + Style.BRIGHT
C_RED = Fore.RED + Style.BRIGHT
C_WHITE = Fore.WHITE + Style.BRIGHT
C_RESET = Style.RESET_ALL

print(f"{C_CYAN}\n[1/3] Carregando dataset e preparando variáveis...{C_RESET}")
x, y, numClass = loadData()
print(f"      {Fore.GREEN}✔ Dataset carregado com sucesso! Total de amostras: {len(x)} | Classes: {numClass}{C_RESET}")

print(f"{C_CYAN}[2/3] Particionando dados (Dirichlet α={ALPHA_DIRICHLET}) e aplicando Split/Scaler...{C_RESET}")
partitions = partitionDataByDirichlet(x, y, numClass)
clients_data = splitData(partitions)
print(f"      {Fore.GREEN}✔ Particionamento concluído para {NUMBER_HOSPITALS} hospitais/clientes.{C_RESET}")

print(f"{C_CYAN}[3/3] Inicializando Modelo Global no Servidor...{C_RESET}")
global_model = create_model(numClass)
reusable_model = create_model(numClass)
global_weights = global_model.get_weights()
print(f"      {Fore.GREEN}✔ Modelo compilado e pronto para o Aprendizado Federado!{C_RESET}")

print(f"\n{C_MAGENTA}{'='*60}")
print(f"{C_MAGENTA}          INICIANDO SIMULAÇÃO DE APRENDIZADO FEDERADO          ")
print(f"{C_MAGENTA}{'='*60}{C_RESET}")

all_final_y_true = []
all_final_y_pred = []

for round_num in range(NUMBER_ROUNDS):

    print(f"\n{C_YELLOW}>>> RODADA {round_num + 1}/{NUMBER_ROUNDS} <<<{C_RESET}")
    print(f"{Fore.WHITE}{'-'*60}{C_RESET}")

    client_weights = []
    client_sizes = []
    client_metrics = []

    for client_id, (x_train, y_train, x_test, y_test) in enumerate(clients_data):

        if len(x_train) == 0:
            print(f"  {C_RED} Hospital {client_id}: Sem dados de treino.{C_RESET}")
            continue

        reusable_model.set_weights(global_weights)

        # Avaliação do Modelo Global no conjunto de teste do cliente
        if len(x_test) > 0:            
            
            y_pred, y_probs = fast_predict(reusable_model, x_test)

            if round_num == NUMBER_ROUNDS - 1:
                all_final_y_true.extend(y_test)
                all_final_y_pred.extend(y_pred)

            # Cálculo de métricas
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            loss_val = log_loss(y_test, y_probs, labels=list(range(numClass)))

            all_class = np.unique(y_test)

            if (len(all_class) > 1):
                y_probs_in = y_probs[:, all_class]
                y_probs_in = y_probs_in / np.sum(y_probs_in, axis=1, keepdims=True)
                
                auc_val = roc_auc_score(y_test, y_probs_in, multi_class='ovr', average='macro', labels=all_class)
            else:
                auc_val = 0.0

            client_metrics.append((len(x_test), {
                "loss": loss_val,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc": auc_val
            }))

            print(f"  {C_BLUE} Hospital {client_id}{C_RESET} | Treino: {len(x_train)} | Teste: {len(x_test)}")
            print(f"     └─ {C_WHITE}Loss: {loss_val:.4f}{C_RESET} | {C_GREEN}Accuracy: {acc:.4f}{C_RESET} | {C_CYAN}Precision: {prec:.4f}{C_RESET} | {C_YELLOW}Recall: {rec:.4f}{C_RESET} | {C_MAGENTA}F1 Score: {f1:.4f}{C_RESET}| {C_CYAN}AUC: {auc_val:.4f}{C_RESET}")

        # Treinamento Local a partir dos pesos Globais
        reusable_model.fit(
            x_train,
            y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        client_weights.append(reusable_model.get_weights())
        client_sizes.append(len(x_train))

    # Agregação com FedAvg
    if len(client_weights) > 0:
        global_weights = federated_average(client_weights, client_sizes)
        global_model.set_weights(global_weights)

    # Exibição do resumo Média Global Ponderada
    if len(client_metrics) > 0:
        total_test_samples = sum(n for n, _ in client_metrics)

        avg_loss = sum(n * m["loss"] for n, m in client_metrics) / total_test_samples
        avg_acc = sum(n * m["accuracy"] for n, m in client_metrics) / total_test_samples
        avg_prec = sum(n * m["precision"] for n, m in client_metrics) / total_test_samples
        avg_rec = sum(n * m["recall"] for n, m in client_metrics) / total_test_samples
        avg_f1 = sum(n * m["f1_score"] for n, m in client_metrics) / total_test_samples
        avg_auc = sum(n * m["auc"] for n, m in client_metrics) / total_test_samples

        print(f"\n  {C_MAGENTA} METRICAS GLOBAIS AGREGADAS (RODADA {round_num + 1}){C_RESET}")
        print(f"  {C_WHITE} Loss Média      : {avg_loss:.4f}{C_RESET}")
        print(f"  {C_GREEN} Acurácia Média  : {avg_acc:.4f}{C_RESET}")
        print(f"  {C_CYAN} Precisão Média  : {avg_prec:.4f}{C_RESET}")
        print(f"  {C_YELLOW} Recall Médio    : {avg_rec:.4f}{C_RESET}")
        print(f"  {C_MAGENTA} F1-Score Médio  : {avg_f1:.4f}{C_RESET}")
        print(f"  {C_CYAN} AUC Média       : {avg_auc:.4f}{C_RESET}")

print(f"\n{C_GREEN}{'='*60}")
print(f"{C_GREEN}✔ SIMULAÇÃO FEDERADA CONCLUÍDA COM SUCESSO!")
print(f"{C_GREEN}{'='*60}{C_RESET}\n")

print(f"{C_CYAN}Gerando e salvando imagem da Matriz de Confusão...{C_RESET}")

confusionMatrix = confusion_matrix(all_final_y_true, all_final_y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Matriz de Confusão - Modelo Global (Última Rodada)')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.tight_layout()

plt.savefig('matriz_confusao.png', dpi=300)
plt.close()

print(f"{C_GREEN}✔ Matriz de Confusão salva com sucesso!{C_RESET}\n")