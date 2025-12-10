from pathlib import Path

def labelData(file):
    fileName = file.stem
    fileToRead = open(str(file), "r", encoding="utf-8")
    modifiedFile = open(f"datasetScript/finalData/{fileName}.txt", "w", encoding="utf-8")

    for line in fileToRead:
        line = line.strip()
        currentLine = round(float(line.split(",")[1]))
    
        if currentLine >= 140:
            modifiedFile.write(line + " Severe Tachycardia \n")
        elif currentLine >= 100:
            modifiedFile.write(line + " Tachycardia \n")
        elif currentLine >= 60:
            modifiedFile.write(line + " Rest \n")
        elif currentLine >= 40:
            modifiedFile.write(line + " Light Bradycardia \n")
        else:   
            modifiedFile.write(line + " Severe Bradycardia \n")

    fileToRead.close()
    modifiedFile.close()
    
for file_path in Path("datasetScript/initialData").glob("*.txt"):
    labelData(file_path)


