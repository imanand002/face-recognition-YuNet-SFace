import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_embeddings(data_dir):
    # This function will load embeddings from files in the specified directory
    embeddings = []
    labels = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            data = np.load(file_path)  # Assuming embeddings are stored in .npy files
            embeddings.append(data['embeddings'])
            labels.append(data['labels'])
    return np.concatenate(embeddings), np.concatenate(labels)

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_classifier(data_dir, epochs=10):
    embeddings, labels = load_embeddings(data_dir)
    model = build_model(embeddings.shape[1:])
    
    model.fit(embeddings, labels, epochs=epochs)

if __name__ == '__main__':
    train_classifier('data/images/', epochs=50)  # Default epochs can be overridden
