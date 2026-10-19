import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from colorama import Fore, Style, init

from config import *
from data import *
from model import *

warnings.filterwarnings("ignore")
init(autoreset=True)

C_CYAN = Fore.CYAN + Style.BRIGHT
C_GREEN = Fore.GREEN + Style.BRIGHT
C_YELLOW = Fore.YELLOW + Style.BRIGHT
C_MAGENTA = Fore.MAGENTA + Style.BRIGHT
C_BLUE = Fore.BLUE + Style.BRIGHT
C_RED = Fore.RED + Style.BRIGHT
C_WHITE = Fore.WHITE + Style.BRIGHT
C_RESET = Style.RESET_ALL

def calculate_auc(y_true, y_probs, num_classes):

    classes = np.unique(y_true)

    if (len(classes) < 2):
        return float("nan")

    if (num_classes == 2):
        return roc_auc_score((y_true == 1).astype(int), y_probs[:, 1])

    probs = y_probs[:, classes]
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    return roc_auc_score(y_true, probs, labels=classes, multi_class="ovr", average="macro")


def calculate_metrics(y_true, y_pred, y_probs, num_classes):

    return {
        "loss": log_loss(y_true, y_probs, labels=np.arange(num_classes)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": calculate_auc(y_true, y_probs, num_classes)
    }


def print_client_metrics(client_id, metrics, client_data):

    x_train, _, x_test, _ = client_data

    print(f"  {C_BLUE}Hospital {client_id + 1}{C_RESET} | Treino: {len(x_train)} | Teste: {len(x_test)}")

    auc = metrics["auc"]
    auc_text = f"{auc:.4f}" if np.isfinite(auc) else "N/A"

    print(
        f"     └─ {C_WHITE}Loss: {metrics['loss']:.4f}{C_RESET} | "
        f"{C_GREEN}Accuracy: {metrics['accuracy']:.4f}{C_RESET} | "
        f"{C_CYAN}Precision: {metrics['precision']:.4f}{C_RESET} | "
        f"{C_YELLOW}Recall: {metrics['recall']:.4f}{C_RESET} | "
        f"{C_MAGENTA}F1 Score: {metrics['f1_score']:.4f}{C_RESET} | "
        f"{C_CYAN}AUC: {auc_text}{C_RESET}"
    )


def print_global_metrics(metrics, round_num):

    print(f"\n  {C_MAGENTA} MÉTRICAS GLOBAIS DO SERVIDOR (RODADA {round_num + 1}){C_RESET}")

    auc = metrics["auc"]
    auc_text = f"{auc:.4f}" if np.isfinite(auc) else "N/A"

    print(f"  {C_WHITE}Loss      : {metrics['loss']:.4f}{C_RESET}")
    print(f"  {C_GREEN}Accuracy  : {metrics['accuracy']:.4f}{C_RESET}")
    print(f"  {C_CYAN}Precision : {metrics['precision']:.4f}{C_RESET}")
    print(f"  {C_YELLOW}Recall    : {metrics['recall']:.4f}{C_RESET}")
    print(f"  {C_MAGENTA}F1 Score  : {metrics['f1_score']:.4f}{C_RESET}")
    print(f"  {C_CYAN}AUC       : {auc_text}{C_RESET}")


def train_local_client(client_data, global_weights, num_classes):

    x_train, y_train, _, _ = client_data

    local_model = create_model(num_classes)
    local_model.set_weights(global_weights)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    if (AGGREGATION_METHOD == 2):
        updated_weights = train_client_fedprox(
            model=local_model, global_weights=global_weights,
            x_train=x_train, y_train=y_train,
            epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE,
            mu=MU_PROXIMAL, class_weights_dict=class_weights_dict
        )
    else:
        local_model.fit(
            x_train, y_train,
            epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE,
            class_weight=class_weights_dict, verbose=0
        )

        updated_weights = local_model.get_weights()

    return updated_weights, len(x_train)


def evaluate_clients(global_model, clients_data, num_classes):

    client_metrics = []

    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    for client_id, client_data in enumerate(clients_data):
        
        _, _, x_test, y_test = client_data

        if (len(x_test) == 0):
            print(f"  {C_RED}Hospital {client_id + 1}: sem dados de teste; ignorado.{C_RESET}")
            continue

        y_pred, y_probs = fast_predict(global_model, x_test)
        metrics = calculate_metrics(y_test, y_pred, y_probs, num_classes)

        print_client_metrics(client_id, metrics, client_data)
        client_metrics.append((len(y_test), metrics))

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_probs.extend(y_probs.tolist())

    if (not all_y_true):
        raise ValueError("Nenhum hospital possui dados de teste.")

    y_true = np.asarray(all_y_true)
    y_pred = np.asarray(all_y_pred)
    y_probs = np.asarray(all_y_probs)

    global_metrics = calculate_metrics(y_true, y_pred, y_probs, num_classes)
    return global_metrics, y_true, y_pred


def main():
    
    print(f"{C_CYAN}\n[1/3] Carregando dataset e preparando variáveis... {C_RESET}")

    x, y, num_classes = loadData()

    print(f"      {C_GREEN}✔ Dataset carregado! Amostras: {len(x)} | Classes: {num_classes}{C_RESET}")
    print(f"{C_CYAN}[2/3] Particionando dados (Dirichlet α={ALPHA_DIRICHLET}) e aplicando Split/Scaler... {C_RESET}")

    partitions = partitionDataByDirichlet(x, y, num_classes)
    clients_data = splitData(partitions)

    print(f"      {C_GREEN}✔ Particionamento concluído para {NUMBER_HOSPITALS} hospitais/clientes.{C_RESET}")
    print(f"\n{C_WHITE}DISTRIBUIÇÃO DOS DADOS POR HOSPITAL {C_RESET}")

    for i, (x_train, y_train, x_test, y_test) in enumerate(clients_data):
        print(f"  Hospital {i + 1} | Treino: {len(x_train)} | Teste: {len(x_test)} | Classes no treino: {len(np.unique(y_train))}")

    method_names = {1: "FedAvg", 2: "FedProx", 3: "FedAdam"}

    if (AGGREGATION_METHOD not in method_names):
        raise ValueError("AGGREGATION_METHOD deve ser 1 (FedAvg), 2 (FedProx) ou 3 (FedAdam).")

    print(f"\n{C_CYAN}[3/3] Inicializando Modelo Global no Servidor...{C_RESET}")

    global_model = create_model(num_classes)
    global_weights = global_model.get_weights()

    server_m = None
    server_v = None

    print(f"      {C_GREEN}✔ Modelo global inicializado.{C_RESET}")
    print(f"\n{C_MAGENTA}{'=' * 65}")
    print("          INICIANDO SIMULAÇÃO DE APRENDIZADO FEDERADO")
    print(f"          Método de agregação: {method_names[AGGREGATION_METHOD]}")
    print(f"{'=' * 65}{C_RESET}")

    final_y_true = None
    final_y_pred = None

    for round_num in range(NUMBER_ROUNDS):
        print(f"\n{C_YELLOW}>>> RODADA {round_num + 1}/{NUMBER_ROUNDS} <<<{C_RESET}")
        print(f"{Fore.WHITE}{'-' * 65}{C_RESET}")

        client_weights = []
        client_sizes = []

        print(f"\n{C_CYAN}TREINAMENTO LOCAL DOS CLIENTES{C_RESET}")

        for client_id, client_data in enumerate(clients_data):
            x_train, y_train, _, _ = client_data

            if (len(x_train) == 0):
                print(f"  {C_RED}Hospital {client_id + 1}: sem dados de treino.{C_RESET}")
                continue

            updated_weights, client_size = train_local_client(client_data, global_weights, num_classes)
            client_weights.append(updated_weights)
            client_sizes.append(client_size)

            print(f"  {C_GREEN}✔ Hospital {client_id + 1}: treinamento concluído ({client_size} amostras).{C_RESET}")

        if (not client_weights):
            raise RuntimeError("Nenhum cliente conseguiu realizar treinamento.")

        print(f"\n{C_CYAN}AGREGAÇÃO NO SERVIDOR{C_RESET}")

        if (AGGREGATION_METHOD == 3):
            global_weights, server_m, server_v = federated_adam(
                global_weights, client_weights, client_sizes,
                server_m, server_v, round_num
            )

        else:
            global_weights = federated_average(client_weights, client_sizes)

        global_model.set_weights(global_weights)

        print(f"  {C_GREEN}✔ Pesos globais atualizados usando {method_names[AGGREGATION_METHOD]}.{C_RESET}")
        print(f"\n{C_CYAN}TESTE DO MODELO GLOBAL EM CADA HOSPITAL{C_RESET}")

        global_metrics, final_y_true, final_y_pred = evaluate_clients(global_model, clients_data, num_classes)
        print_global_metrics(global_metrics, round_num)

    print(f"\n{C_GREEN}{'=' * 65}")
    print("✔ SIMULAÇÃO FEDERADA CONCLUÍDA COM SUCESSO!")
    print(f"{'=' * 65}{C_RESET}")
    print(f"\n{C_CYAN}Gerando e salvando imagem da Matriz de Confusão do Modelo Global Final...{C_RESET}")

    show_confusion_matrix(final_y_true, final_y_pred)

    print(f"{C_GREEN}✔ Matriz de Confusão salva em matriz_confusao.png{C_RESET}\n")

if __name__ == "__main__":
    main()