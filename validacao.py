from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Carregar o modelo
modelo = load_model('modelo_lesoes_completo.h5')

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
    percentual, classe = modelo.predict(img_array)
    
    # Converter resultados
    percentual_lesao = percentual[0][0]
    classe_nome = le.inverse_transform([np.argmax(classe[0])])[0]
    
    return percentual_lesao, classe_nome

# Exemplo de uso
caminho_imagem = 'validacao/sua_imagem.png'
percentual, classe = prever_imagem(caminho_imagem)
print(f'Percentual de lesão: {percentual:.2f}%')
print(f'Classe: {classe}')