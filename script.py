import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Ler o arquivo Excel
df = pd.read_excel('Dataset.xlsx')  
print(df.head())  

# Adicionar a extensão .png aos nomes dos arquivos
df['folha'] = df['folha'] + '.png'

# Supondo que seu DataFrame tenha as colunas 'nome_imagem' e 'percentual'
# Certifique-se de que os nomes das colunas correspondem aos do seu arquivo Excel.

# 2. Dividir os dados em treino e validação
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# 3. Configurar o ImageDataGenerator para pré-processamento
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalização dos pixels
    rotation_range=20,       # Aumenta a variabilidade com rotações
    horizontal_flip=True,    # Flip horizontal
    vertical_flip=True       # (Opcional) Flip vertical, se fizer sentido para as imagens
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 4. Criar os geradores a partir do DataFrame
# Supondo que as imagens estão na pasta 'imagens'
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='dataset',        # Pasta onde estão as imagens
    x_col='folha',        # Coluna com os nomes dos arquivos
    y_col='percentual',         # Coluna com o rótulo (valor percentual de lesão)
    target_size=(224, 224),     # Redimensionamento para o tamanho esperado pelo modelo
    batch_size=32,
    class_mode='raw'            # 'raw' para regressão; se for classificação, use 'categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='dataset',
    x_col='folha',
    y_col='percentual',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'            # Alterado para 'raw' para regressão
)

# Agora, o pipeline está pronto para ser integrado ao seu modelo, como mostrado anteriormente.
