import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# 1. Data Loading and Preparation
data_dir = 'path/to/your/dataset'  # Replace with your dataset path
image_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split for validation
)

# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )

# # 2. Model Building (Simple CNN)
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Binary classification
# ])

# # 3. Model Compilation
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # 4. Model Training
# epochs = 20
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )

# # 5. Model Evaluation (using a separate test set if available)
# # Assuming you have a 'test_dir'
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     'path/to/your/test_data',  # Replace with your test data path
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False  # Important for getting correct predictions order
# )

# predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)
# predicted_classes = (predictions > 0.5).astype(int)
# true_classes = test_generator.classes
# class_labels = list(test_generator.class_indices.keys())

# print("Classification Report:")
# print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# cm = confusion_matrix(true_classes, predicted_classes)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()