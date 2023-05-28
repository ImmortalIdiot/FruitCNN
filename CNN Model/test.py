import keras.models
from keras.preprocessing.image import image_utils
import numpy as np
import matplotlib.pyplot as plt
import os
classes = os.listdir('fullFruitDataset/Training')

IMAGE_SIZE = 100
BATCH_SIZE = 8

model = keras.models.load_model("Model 7.1")
x = 0
path = r"test pics"

for i in os.listdir(path):
    img = image_utils.load_img(f"test pics/{i}", target_size=(IMAGE_SIZE, IMAGE_SIZE))

    image_tensor = image_utils.img_to_array(img)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor /= 255

    prediction = model.predict(image_tensor)

    plt.subplots(1, 1)
    plt.imshow(img)
    plt.xlabel(classes[np.argmax(prediction)])
    plt.grid(False)
    x += 1

plt.show()
