import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Створюємо просту послідовну модель
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # для бінарної класифікації
])

# 2. Компілюємо модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Генеруємо випадкові дані для навчання
import numpy as np
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, size=(1000, 1))

# 4. Навчання моделі
model.fit(X, y, epochs=5, batch_size=32)
