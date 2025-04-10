import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

train_dir = "skin-cancer-images/train"  # Replace with your actual path
test_dir = "skin-cancer-images/test"  # Replace with your actual path
image_size = (128, 128)  # Reduced image size to speed up processing
batch_size = 32
epochs = 50
class_names = ["benign", "malignant"]

def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random

    random.seed(seed)

set_seeds()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,  # Reduced augmentation
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    seed=42,
    classes=class_names,
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    seed=42,
    classes=class_names,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
    seed=42,
    classes=class_names,
)

model = Sequential(
    [
        tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(16, (3, 3), activation="relu", padding='same'),  # Reduced filters
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu", padding='same'),  # Reduced filters
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding='same'),  # Reduced filters
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),  # Reduced units
        Dropout(0.3),  # Reduced dropout
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True  # Adjusted patience
)
model_checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_accuracy", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks,
    verbose=1,
)

test_steps = test_generator.samples // test_generator.batch_size
if test_generator.samples % test_generator.batch_size != 0:
    test_steps += 1

predictions = model.predict(test_generator, steps=test_steps, verbose=1)
predicted_probabilities = predictions.flatten()
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("Classification Report (Test Data):")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Data)")
plt.show()

fpr, tpr, thresholds = roc_curve(true_classes, predicted_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy vs. Epoch")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss vs. Epoch")
plt.show()

model.save("skin_cancer_model.h5")
print("Model saved to skin_cancer_model.h5")