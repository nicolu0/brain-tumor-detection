import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil

from tensorflow import keras
from keras import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

df = pd.read_csv("Brain Tumor.csv")[["Image", "Class"]]
# print(df)
data = df.copy()
train_data, val_data, test_data = np.split(
    data.sample(frac=1, random_state=42), [int(len(data) * 0.8), int(len(data) * 0.9)]
)
# print(df_shuffle)

# part1 = int(len(df_shuffle) * 0.8)
# part2 = int(len(df_shuffle) * 0.9)

# train_data = df_shuffle[:part1]
# val_data = df_shuffle[part1:part2]
# test_data = df_shuffle[part2:]

# print(train_data)
# print(val_data)
# print(test_data)

dir = "../brain_tumor_detection/data"


# Creates train, val, test directories
def create_dirs(base_dir):
    path = os.path.join(base_dir, "model")
    if os.path.isdir(path):
        shutil.rmtree(path)
        print("Removed directories.")

    for label in ["0", "1"]:
        train_dir = os.path.join(base_dir, "model", "train", label)
        val_dir = os.path.join(base_dir, "model", "validation", label)
        test_dir = os.path.join(base_dir, "model", "test", label)

        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)


# Adds images from train, val, test dataframes to their respective directories
def add_jpg_to_dirs(base_dir):
    for i in train_data.index:
        if train_data["Class"][i] == 0:
            filename = f"{train_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "train", "0", filename)
            shutil.copyfile(source, dest)
        elif train_data["Class"][i] == 1:
            filename = f"{train_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "train", "1", filename)
            shutil.copyfile(source, dest)

    for i in val_data.index:
        if val_data["Class"][i] == 0:
            filename = f"{val_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "validation", "0", filename)
            shutil.copyfile(source, dest)
        elif val_data["Class"][i] == 1:
            filename = f"{val_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "validation", "1", filename)
            shutil.copyfile(source, dest)

    for i in test_data.index:
        if test_data["Class"][i] == 0:
            filename = f"{test_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "test", "0", filename)
            shutil.copyfile(source, dest)
        elif test_data["Class"][i] == 1:
            filename = f"{test_data['Image'][i]}.jpg"
            source = os.path.join("Brain Tumor", "Brain Tumor", filename)
            dest = os.path.join(base_dir, "model", "test", "1", filename)
            shutil.copyfile(source, dest)


img_size = 224
size = (img_size, img_size)
shape = (img_size, img_size, 3)


def image_augmentation():
    train_dir = os.path.join(dir, "model", "train")
    val_dir = os.path.join(dir, "model", "validation")
    test_dir = os.path.join(dir, "model", "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=size, batch_size=32, class_mode="binary"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=size, batch_size=32, class_mode="binary"
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=size, batch_size=32, class_mode="binary"
    )

    return train_generator, val_generator, test_generator


train_generator, val_generator, test_generator = image_augmentation()


def learning_model():
    model = Sequential()
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=shape, include_top=False, weights="imagenet"
    )

    pretrained_model.trainable = False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.6))
    # model.add(Dense(1024, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.build((None, img_size, img_size, 3))
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_generator, validation_data=val_generator, epochs=10)

    model.save("../brain_tumor_detection/my_model")
    return model


# create_dirs(dir)
# add_jpg_to_dirs(dir)

# model = learning_model()
