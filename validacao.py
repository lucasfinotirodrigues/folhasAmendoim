from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Carregar o modelo
modelo = load_model('modelo_lesoes_folhas.h5')

# Carregar o LabelEncoder com as mesmas classes do treinamento
df = pd.read_excel('Dataset.xlsx')
le = LabelEncoder()
le.fit(df['Classe Moderada'])

def prever_imagem(caminho_imagem):
    # Preparar a imagem
    img = load_img(caminho_imagem, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Fazer predições
    percentual = modelo.predict(img_array)[0][0]
    
    # Determinar a classe e retornar os resultados
    return determinar_classe(percentual)

def determinar_classe(percentual):
    if percentual <= 0:
        classe = "Sadia"
    elif 0.001 <= percentual <= 0.02:
        classe = "Muito Leve"
    elif 0.021 <= percentual <= 0.09:
        classe = "Leve"
    elif 0.091 <= percentual <= 0.12:
        classe = "Moderado"
    elif 0.121 <= percentual <= 0.36:
        classe = "Severo"
    elif percentual > 0.36:
        classe = "Muito Severo"
    
    return percentual, classe


# Exemplo de uso
caminho_imagem = 'validacao/img 1 - img 1.png'
# caminho_imagem = 'validacao/img 2 - img 5.png'
# caminho_imagem = 'validacao/img 3 - img 7.png'
# caminho_imagem = 'validacao/img 4 - img 15.png'
# caminho_imagem = 'validacao/img 5 - img-60.png'
# caminho_imagem = 'validacao/img 6 - img 77.png'
# caminho_imagem = 'validacao/img 7 - img 98.png'
# caminho_imagem = 'validacao/img 8 - img 183.png'
# caminho_imagem = 'validacao/img 9 - img 126.png'
# caminho_imagem = 'validacao/img 10 - img 148.png'
# caminho_imagem = 'validacao/img 11 - img 125.png'
# caminho_imagem = 'validacao/img 12 - img 159.png'
# caminho_imagem = 'validacao/img 13 - img 164.png'
# caminho_imagem = 'validacao/img 14 - img 182.png'
# caminho_imagem = 'validacao/img 15 - img 106.png'
# caminho_imagem = 'validacao/img 16 - img 211.png'
# caminho_imagem = 'validacao/img 17 - img 212.png'
# caminho_imagem = 'validacao/img 18 - img 214.png'
# caminho_imagem = 'validacao/img 19 - img 231.png'
# caminho_imagem = 'validacao/img 20 - img 233.png'
# caminho_imagem = 'validacao/img 21 - img 235.png'
# caminho_imagem = 'validacao/img 22 - img 252.png'
# caminho_imagem = 'validacao/img 23 - img 273.png'
# caminho_imagem = 'validacao/img 24 - img 278.png'
# caminho_imagem = 'validacao/img 25 - img 279.png'
# caminho_imagem = 'validacao/img 26 - img 301.png'
# caminho_imagem = 'validacao/img 27 - img 353.png'
# caminho_imagem = 'validacao/img 28 - img 355.png'
# caminho_imagem = 'validacao/img 29 - img 372.png'
# caminho_imagem = 'validacao/img 30 - img 360.png'
percentual, classe = prever_imagem(caminho_imagem)
print(f'Percentual de lesão: {percentual:.2f}%')
print(f'Classe estimada: {classe}')