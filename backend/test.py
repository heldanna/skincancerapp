import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale pixel values (if needed)
    return img_array

model_path = 'skin_cancer_model.h5'  # Replace with your model path
model = load_model(model_path)

image_path = 'skin-cancer-images/test/malignant/44.jpg'  # Replace with a test image path
target_size = (224, 224)  # Must match your training image size

processed_image = load_and_preprocess_image(image_path, target_size)
print("Processed Image Shape:", processed_image.shape)  # Debugging
predictions = model.predict(processed_image)

print("Raw Predictions:", predictions)  # Debugging

# --- Simple Classification (Adjust Threshold as Needed) ---
if predictions[0][0] > 0.5:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")

# --- More Robust Evaluation (for a set of test images) ---
# (You'll need to adapt this if you're testing on a single image)
# test_images_paths = ['path/to/test1.jpg', 'path/to/test2.jpg', ...]  # List of test image paths
# true_labels = [1, 0, ...]  # 1 for malignant, 0 for benign (example)
#
# predicted_probabilities = []
# for img_path in test_images_paths:
#     processed_img = load_and_preprocess_image(img_path, target_size)
#     pred = model.predict(processed_img)
#     predicted_probabilities.append(pred[0][0])
#
# # --- ROC Curve (for binary classification) ---
# fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
#
# # --- Choose a Threshold Based on ROC Curve (Example) ---
# chosen_threshold = 0.4  # Replace with your chosen threshold
# predicted_classes = [1 if prob > chosen_threshold else 0 for prob in predicted_probabilities]
#
# print(classification_report(true_labels, predicted_classes))
# cm = confusion_matrix(true_labels, predicted_classes)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.show()