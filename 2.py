import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import numpy as np

# Створюємо просту модель
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Генеруємо випадкові дані
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, size=(1000, 1))

# Створюємо логгер для TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Навчаємо модель з логуванням
model.fit(X, y, epochs=5, batch_size=32, callbacks=[tensorboard_callback])