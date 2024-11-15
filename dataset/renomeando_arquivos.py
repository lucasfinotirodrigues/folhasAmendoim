import os

def renomear_imagens(diretorio_atual):
    # Listar todos os arquivos no diretório
    arquivos = os.listdir(diretorio_atual)
    contador = 1

    # Filtrar apenas os arquivos de imagem
    extensoes_validas = ['.jpg']
    imagens = [arquivo for arquivo in arquivos if os.path.splitext(arquivo)[1].lower() in extensoes_validas]

    for imagem in imagens:
        extensao = os.path.splitext(imagem)[1]  # Obter a extensão do arquivo
        novo_nome = f"mancha-tardia{contador}{extensao}"
        caminho_antigo = os.path.join(diretorio_atual, imagem)
        caminho_novo = os.path.join(diretorio_atual, novo_nome)
        
        # Renomear o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f"Renomeado: {imagem} -> {novo_nome}")
        contador += 1

# Chamar a função no diretório atual
diretorio_atual = os.getcwd()
renomear_imagens(diretorio_atual)
