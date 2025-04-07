import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# 1. Завантажуємо зображення з Інтернету
url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'  # мопс 
response = requests.get(url)
img = Image.open(BytesIO(response.content)).resize((224, 224))

# 2. Візуалізація
plt.imshow(img)
plt.axis('off')
plt.title("Зображення для класифікації")
plt.show()

# 3. Підготовка зображення
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# 4. Завантаження моделі та передбачення
model = MobileNetV2(weights='imagenet')
predictions = model.predict(img_array)

# 5. Декодуємо результат
decoded = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, name, score) in enumerate(decoded):
    print(f"{i+1}) {name}: {score:.2%}")
