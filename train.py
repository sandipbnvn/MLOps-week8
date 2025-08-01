'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
import sys
import os
import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback
import mlflow
import mlflow.tensorflow

# Fix Unicode issues on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with poisoning experiment')
    parser.add_argument('--p', type=float, default=0.0, help='Poisoning percentage')
    return parser.parse_args()

args = parse_args()

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'model.weights.h5'
train_data_dir = os.path.join('data', 'train')
validation_data_dir = os.path.join('data', 'validation')
cats_train_path = os.path.join(path, train_data_dir, 'cats')
nb_train_samples = 2 * len([name for name in os.listdir(cats_train_path)
                            if os.path.isfile(
                                os.path.join(cats_train_path, name))])
nb_validation_samples = 800
epochs = 10
batch_size = 10


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2)))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (int(nb_validation_samples / 2)) +
        [1] * (int(nb_validation_samples / 2)))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Start MLflow run with error handling
    try:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("poisoning_percentage", args.p)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            # Train model
            history = model.fit(train_data, train_labels,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(validation_data, validation_labels),
                              verbose=0,
                              callbacks=[TqdmCallback(), CSVLogger("metrics.csv")])
            
            # Log metrics
            for epoch in range(epochs):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            # Log metrics file
            mlflow.log_artifact("metrics.csv")
            
            print(f"MLflow run completed successfully for poisoning level {args.p}%")
            
    except Exception as e:
        print(f"MLflow error (continuing without MLflow): {e}")
        # Fallback: train without MLflow
        history = model.fit(train_data, train_labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(validation_data, validation_labels),
                          verbose=0,
                          callbacks=[TqdmCallback(), CSVLogger("metrics.csv")])
    
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model() 