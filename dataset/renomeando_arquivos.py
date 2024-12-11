import os

def renomear_imagens(diretorio_atual):
    arquivos = os.listdir(diretorio_atual)
    contador = 1

    extensoes_validas = ['.jpg']
    imagens = [arquivo for arquivo in arquivos if os.path.splitext(arquivo)[1].lower() in extensoes_validas]

    for imagem in imagens:
        extensao = os.path.splitext(imagem)[1] 
        novo_nome = f"mancha-tardia{contador}{extensao}"
        caminho_antigo = os.path.join(diretorio_atual, imagem)
        caminho_novo = os.path.join(diretorio_atual, novo_nome)
        
        os.rename(caminho_antigo, caminho_novo)
        print(f"Renomeado: {imagem} -> {novo_nome}")
        contador += 1

diretorio_atual = os.getcwd()
renomear_imagens(diretorio_atual)
