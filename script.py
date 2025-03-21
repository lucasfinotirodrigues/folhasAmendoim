import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

# 5. Importar as bibliotecas necessárias para o modelo
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# 6. Criar o modelo base (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 7. Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# 8. Adicionar camadas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='linear')(x)  # Uma saída para regressão

# 9. Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# 10. Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# 11. Treinar o modelo
steps_per_epoch = len(train_df) // 32  # batch_size é 32
validation_steps = len(val_df) // 32    # batch_size é 32

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# 12. Salvar o modelo
model.save('modelo_lesoes_folhas.h5')

# 13. Plotar o histórico de treinamento
plt.figure(figsize=(12, 4))

# Plot do erro (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Erro do Modelo')
plt.xlabel('Época')
plt.ylabel('Erro (MSE)')
plt.legend()

# Plot do erro absoluto médio
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Treino')
plt.plot(history.history['val_mean_absolute_error'], label='Validação')
plt.title('Erro Absoluto Médio')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
