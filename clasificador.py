#%%
# Librerias 
import keras
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# Importar CIFAR10 (DataBase)
from keras.datasets import cifar10

# Separar datos y labels de test y train
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Shape
x_train[[0]]
len(x_train)
x_train.shape
# %%

# Visualización de imágenes
fig = plt.figure(figsize=(30,10))
for i in range(36):
    ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
# %%

# Normalizar intensidad de [0,255] a [0,1]
x_train = x_train/255
x_test = x_test/255

# One-hot encoding
num_classes = len(np.unique(y_train))
y_train =keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Separo los datos de train en datos de train y validación a partir de un punto arbitrario
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

len(x_train)
x_train.shape

# Verifico la forma de datos
print('Forma de x_train:', x_train.shape)

# Datos tengo en cada dataset
print(x_train.shape[0], 'datos de entrenamiento')
print(x_test.shape[0], 'datos de test')
print(x_valid.shape[0], 'datos de validación')
# %%

# Red Neuronal
# Modelo secuencial
model = Sequential()
# 1. Capa de entrada (Flatten de 3D a 1D)
model.add(Flatten(input_shape=(32,32,3)))  
# 2. Capa densa de 1000 nodos y activación ReLU
model.add(Dense(1000, activation='relu'))
# 3. Capa de dropout con factor 0.2
model.add(Dropout(0.2))
# 4. Capa densa de 512 nodos y activación ReLU
model.add(Dense(512, activation='relu'))
# 5. Capa de dropout con factor 0.2
model.add(Dropout(0.2))
# 6. Capa densa con la cantidad de nodos a la salida y función de activación softmax
model.add(Dense(10, activation='softmax')) 

# Resumen del modelo
model.summary()
# %%

# Copilación y Entrenamiento
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# %%

# Callback 'checkpointer' (guardará los mejores pesos en cada iteración)
checkpointer = ModelCheckpoint(filepath='best_model.keras', verbose=1, save_best_only=True)
# Entrenamiento del modelo
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpointer])
# %%

# Evaluar en conjunto de entrenamiento y prueba
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print(f'Precisión en el conjunto de entrenamiento: {train_acc}')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Cargar los mejores pesos del modelo
model.load_weights('best_model.keras')

# Evaluar el modelo en el conjunto de datos de entrenamiento
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print(f'Precisión en el conjunto de entrenamiento: {train_acc}')

# Evaluar el modelo en el conjunto de datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Visualización de métricas de entrenamiento
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Gráficos de precisión
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Gráficos de pérdida
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
#%%

# Etiquetas de CIFAR10
#0: Avión
#1: Automóvil
#2: Pájaro
#3: Gato
#4: Ciervo
#5: Perro
#6: Rana
#7: Caballo
#8: Barco
#9: Camión

# Probemos algunas imagenes de train
# Cargar los mejores pesos del modelo
model.load_weights('best_model.keras')

# Elegir la imagen
image_index = 9
image = x_train[image_index]
true_label = np.argmax(y_train[image_index])

# Preprocesar la imagen para hacer la predicción
image = image.reshape(1, 32, 32, 3) 
image = image / 255.0 

# Hacer la predicción
prediction = model.predict(image)
predicted_label = np.argmax(prediction)

# Mostrar la imagen y la predicción
plt.figure(figsize=(2, 2)) 
plt.imshow(np.squeeze(x_train[image_index]), interpolation='nearest') 
plt.axis('off')
plt.title(f'Predicción: {predicted_label}, Verdadero: {true_label}')
plt.show()
# %%
