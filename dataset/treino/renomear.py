import os

def rename_images():
    # Obtém a lista de arquivos no diretório atual
    files = [f for f in os.listdir() if f.lower().endswith('.png')]
    
    # Ordena os arquivos para manter uma sequência consistente
    files.sort()
    
    # Renomeia os arquivos
    for index, file in enumerate(files, start=1):
        new_name = f"img-{index}.png"
        os.rename(file, new_name)
        print(f"{file} -> {new_name}")

if __name__ == "__main__":
    rename_images()
