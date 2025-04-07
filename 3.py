import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Завантаження датасету MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Нормалізація зображень
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Створення моделі
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),        # розгортання зображення в вектор
    layers.Dense(128, activation='relu'),        # прихований шар
    layers.Dense(10, activation='softmax')       # 10 класів на виході
])

# 4. Компіляція
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Навчання моделі
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# 6. Оцінка
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Точність на тестових даних: {test_acc:.4f}")
