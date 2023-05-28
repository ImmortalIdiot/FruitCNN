import tensorflow as tf
from tensorflow import keras
import os

IMAGE_SIZE = 100
BATCH_SIZE = 24
TRAIN_PATH = "fruitDataset/Training"
TEST_PATH = "fruitDataset/Test"

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255., rotation_range=15, width_shift_range=0.2,
                                                             height_shift_range=0.2, zoom_range=0.2,
                                                             horizontal_flip=True, fill_mode='nearest')
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)


train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse')

test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=BATCH_SIZE, shuffle=False, class_mode='sparse')

classes_train = os.listdir(TRAIN_PATH)
classes_test = os.listdir(TEST_PATH)

augmentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=911, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.RandomRotation(0.02, seed=911, fill_mode="reflect"),
    tf.keras.layers.RandomContrast(0.2, seed=911),
])

model = tf.keras.models.Sequential([

    augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(len(classes_train), activation="softmax")
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=test_generator, epochs=3)
model.save("Model 2.1")
