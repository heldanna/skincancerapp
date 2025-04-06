import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Optional, but often useful for image handling

def load_and_preprocess_image(image_path, target_size):
    """Loads, resizes, and preprocesses a single image for prediction.

    Args:
        image_path: Path to the image file.
        target_size: Tuple (height, width) specifying the model's expected input size.

    Returns:
        A preprocessed NumPy array representing the image, or None if there's an error.
    """
    try:
        # Using cv2 for more robust image loading
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (if needed)
        img = cv2.resize(img, target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale pixel values (if you did this during training)
        return img_array

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while loading/preprocessing the image: {e}")
        return None


def predict_single_image(model_path, image_path, target_size, class_names):
    """Loads a trained model and predicts the class of a single image.

    Args:
        model_path: Path to the saved model file (.h5 or .keras).
        image_path: Path to the image file to predict.
        target_size: Tuple (height, width) specifying the model's expected input size.
        class_names: List of class names (e.g., ["benign", "malignant"]).

    Returns:
        A dictionary containing the prediction results, or None if there's an error.
    """

    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: Model file not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

    processed_image = load_and_preprocess_image(image_path, target_size)

    if processed_image is None:
        return None

    prediction = model.predict(processed_image)
    probability_malignant = float(prediction[0][0])  # Probability of being malignant
    probability_benign = 1.0 - probability_malignant

    results = {
        class_names[0]: probability_benign,
        class_names[1]: probability_malignant,
    }

    return results


if __name__ == "__main__":
    # --- Example Usage ---
    model_path = "skin_cancer_model.h5"  # Replace with the actual path to your saved model
    image_path = "skin-cancer-images/test/benign/18.jpg"  # Replace with the path to your test image
    target_size = (128, 128)  # Must match your training image size
    class_names = ["benign", "malignant"]  # Ensure this matches your training order

    # --- Load and Preprocess Image ---
    processed_image = load_and_preprocess_image(image_path, target_size)

    if processed_image is not None:
        # --- Make Prediction ---
        prediction_results = predict_single_image(model_path, image_path, target_size, class_names)

        if prediction_results:
            print("Prediction Results:")
            for class_name, probability in prediction_results.items():
                print(f"  {class_name}: {probability:.4f}")

            # --- Simple Classification (Adjust Threshold as Needed) ---
            predicted_class = max(prediction_results, key=prediction_results.get)
            print(f"Predicted Class: {predicted_class}")

            # --- Visualize (Optional) ---
            try:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title(f"Prediction: {predicted_class}")
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"Error displaying image: {e}")

        else:
            print("Prediction failed.")

    else:
        print("Image processing failed.")