import pandas as pd
from pathlib import Path

def label_data(file_path):
    # Extrai o nome do arquivo sem a extensão
    file_name = file_path.stem

    # Lê o arquivo
    df = pd.read_csv(file_path, header=None, names=["Time", "Value"])

    # Adiciona a coluna "Label" com base na coluna "Value"
    df["Label"] = df["Value"].round().apply(lambda x: "Physical Activity" if x>= 120 else "Rest")

    # Salva o DataFrame modificado em um novo arquivo
    output_path = f"hcpa-wearable-monitoring/datasetScript/finalData/{file_name}.txt"

    # Salva sem cabeçalho e sem índice, como no arquivo original
    df.to_csv(output_path, header=False, index=False)

# Processa todos os arquivos na pasta inicialData
for file_path in Path("hcpa-wearable-monitoring/datasetScript/initialData").glob("*.txt"):
    label_data(file_path)