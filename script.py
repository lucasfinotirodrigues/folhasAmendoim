import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Ler o arquivo Excel
df = pd.read_excel('Dataset.xlsx')  
print(df.head())  

# Adicionar a extensão .png aos nomes dos arquivos
df['folha'] = df['folha'] + '.png'

# 2. Converter as classes em números (one-hot encoding)
le = LabelEncoder()
df['classe_num'] = le.fit_transform(df['Classe Moderada'])

# 3. Dividir os dados em treino e validação
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# 4. Configurar o ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 5. Criar os geradores
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='dataset',
    x_col='folha',
    y_col='percentual',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='dataset',
    x_col='folha',
    y_col='percentual',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

# 6. Criar o modelo com duas saídas
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar camadas base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas compartilhadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# Camada para percentual
output_percentual = Dense(1, activation='linear', name='percentual')(x)

# Criar modelo
model = Model(inputs=base_model.input, outputs=output_percentual)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Treinar modelo
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    steps_per_epoch=len(train_df) // 32,
    validation_steps=len(val_df) // 32
)

# Salvar modelo
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
