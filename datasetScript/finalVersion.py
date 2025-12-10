from pathlib import Path

def labelData(file):
    #Pega só o nome do arquivo sem a extensão
    fileName = file.stem
    #Abre o arquivo para leitura
    fileToRead = open(str(file), "r", encoding="utf-8")
    #Cria um novo arquivo para escrita dos dados modificados
    modifiedFile = open(f"datasetScript/finalData/{fileName}.txt", "w", encoding="utf-8")

    #Itera em cada linha do arquivo original
    for line in fileToRead:
        #Remove espaços em branco
        line = line.strip()
        #Pega o valor da segunda coluna(batimentos cardíacos)
        currentLine = round(float(line.split(",")[1]))

        #Verifica se o valor é maior ou igual a 120
        if currentLine >= 120:
            modifiedFile.write(line + " Physical Activity \n")
        else: 
            modifiedFile.write(line + " Rest \n")

    #Fecha os arquivos
    fileToRead.close()
    modifiedFile.close()
    
#Itera em todos os arquivos .txt na pasta initialData
for file_path in Path("datasetScript/initialData").glob("*.txt"):
    labelData(file_path)


