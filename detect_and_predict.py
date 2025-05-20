import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from colorama import Fore, Style, init as colorama_init
import random
import tensorflow as tf

# Set random seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

colorama_init()

# --- CONFIG ---
IMG_SIZE = 224
TRAIN_DIR = "train-data"
UNLABELED_DIR = "image-data"
MODEL_PATH = "deepfake_model.h5"
EPOCHS = 30  # Increased epochs
BATCH_SIZE = 32  # Increased batch size
LEARNING_RATE = 1e-4  # Adjusted learning rate


# --- LOAD TRAINING DATA ---
def load_labeled_data():
    data = []
    labels = []
    class_counts = {"real": 0, "fake": 0}

    print(f"{Fore.CYAN}Loading training data...{Style.RESET_ALL}")

    for class_name in ["real", "fake"]:
        folder = os.path.join(TRAIN_DIR, class_name)
        label = 0 if class_name == "real" else 1

        if not os.path.exists(folder):
            print(f"{Fore.RED}Error: {folder} directory doesn't exist!{Style.RESET_ALL}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        class_counts[class_name] = len(files)

        for file in files:
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"{Fore.YELLOW}Warning: Couldn't read {img_path}{Style.RESET_ALL}")
                    continue

                # Convert BGR to RGB (important for EfficientNet)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Face detection (optional but improves accuracy)
                # img = detect_and_crop_face(img)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img_to_array(img)
                img = preprocess_input(img)
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"{Fore.RED}Error processing {img_path}: {e}{Style.RESET_ALL}")

    print(
        f"{Fore.GREEN}Loaded {class_counts['real']} real images and {class_counts['fake']} fake images{Style.RESET_ALL}")

    if not data:
        raise ValueError("No valid training images found!")

    X = np.array(data, dtype="float32")
    y = to_categorical(np.array(labels), num_classes=2)
    return X, y


# --- Helper function to detect faces (optional enhancement) ---
def detect_and_crop_face(image):
    # This is a placeholder - you can implement actual face detection here
    # using OpenCV's haarcascade or dlib's face detector
    # For now, we just return the original image
    return image


# --- BUILD MODEL ---
def build_model():
    # Use EfficientNetB0 as base model with explicit input shape
    # This approach avoids the UserWarning about tensor structure
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Freeze early layers to prevent overfitting on small datasets
    for layer in base.layers[:100]:
        layer.trainable = False

    # Add custom classification head
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


# --- TRAIN MODEL ---
def train_model():
    print(f"{Fore.CYAN}Starting training process...{Style.RESET_ALL}")

    try:
        X, y = load_labeled_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        print(f"Training set: {X_train.shape[0]} images")
        print(f"Testing set: {X_test.shape[0]} images")

        # Data augmentation to help prevent overfitting
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Build and train model
        model = build_model()

        # Callbacks for better training
        callbacks = [
            ModelCheckpoint(
                MODEL_PATH,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]

        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            callbacks=callbacks
        )

        # Evaluate model
        print(f"\n{Fore.CYAN}Evaluating model...{Style.RESET_ALL}")
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0, 1], ['Real', 'Fake'])
        plt.yticks([0, 1], ['Real', 'Fake'])

        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig("training_history.png")

        print(f"\n{Fore.GREEN}✅ Model trained and saved to {MODEL_PATH}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Training plots saved to training_history.png and confusion_matrix.png{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error during training: {e}{Style.RESET_ALL}")
        raise


# --- PREDICT IMAGES ---
def predict_unlabeled():
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"{Fore.RED}Error: Model file not found at {MODEL_PATH}{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Loading model from {MODEL_PATH}...{Style.RESET_ALL}")
        model = load_model(MODEL_PATH)

        if not os.path.exists(UNLABELED_DIR):
            print(f"{Fore.RED}Error: Unlabeled data directory not found at {UNLABELED_DIR}{Style.RESET_ALL}")
            return

        files = [f for f in os.listdir(UNLABELED_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        if not files:
            print(f"{Fore.YELLOW}Warning: No image files found in {UNLABELED_DIR}{Style.RESET_ALL}")
            return

        results = []
        confidence_scores = []

        print(f"\n{Fore.CYAN}📂 Analyzing {len(files)} images from {UNLABELED_DIR}...{Style.RESET_ALL}\n")

        for filename in files:
            filepath = os.path.join(UNLABELED_DIR, filename)
            try:
                img = cv2.imread(filepath)
                if img is None:
                    print(f"{Fore.YELLOW}Warning: Couldn't read {filepath}{Style.RESET_ALL}")
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Optional face detection step
                # img = detect_and_crop_face(img)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img_to_array(img)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                # Get prediction
                prediction = model.predict(img, verbose=0)[0]
                fake_prob = prediction[1]  # Probability of being fake
                confidence = max(prediction)
                label = "Fake" if fake_prob > 0.5 else "Real"

                # Calculate confidence adjusted for decision threshold
                adjusted_confidence = fake_prob if label == "Fake" else (1 - fake_prob)
                confidence_scores.append(adjusted_confidence)

                results.append({
                    "filename": filename,
                    "prediction": label,
                    "confidence": float(adjusted_confidence)
                })

                # Enhanced Terminal Output
                color = Fore.RED if label == "Fake" else Fore.GREEN
                bar_len = int(adjusted_confidence * 20)
                bar = f"{'█' * bar_len}{' ' * (20 - bar_len)}"
                print(f"{color}{filename:<40} | {label:^5} | Conf: {adjusted_confidence:.4f} | {bar}{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}❌ Error with {filename}: {e}{Style.RESET_ALL}")

        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv("prediction_results.csv", index=False)

            # Generate statistics
            real_count = df[df["prediction"] == "Real"].shape[0]
            fake_count = df[df["prediction"] == "Fake"].shape[0]
            total = real_count + fake_count
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            print(f"\n{Fore.CYAN}🔍 Summary:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Real: {real_count} ({real_count / total * 100:.1f}%){Style.RESET_ALL}")
            print(f"{Fore.RED}❌ Fake: {fake_count} ({fake_count / total * 100:.1f}%){Style.RESET_ALL}")
            print(f"📊 Average confidence: {avg_confidence:.4f}")
            print(f"📊 Results saved to {Fore.YELLOW}prediction_results.csv{Style.RESET_ALL}")

            # Save summary to file
            with open("prediction_summary.txt", "w") as f:
                f.write(f"Total images analyzed: {total}\n")
                f.write(f"Real images: {real_count} ({real_count / total * 100:.1f}%)\n")
                f.write(f"Fake images: {fake_count} ({fake_count / total * 100:.1f}%)\n")
                f.write(f"Average confidence: {avg_confidence:.4f}\n")

                # Add the top 5 most confidently detected fakes
                top_fakes = df[df["prediction"] == "Fake"].sort_values("confidence", ascending=False).head(5)
                if not top_fakes.empty:
                    f.write("\nTop 5 most confident fake detections:\n")
                    for _, row in top_fakes.iterrows():
                        f.write(f"- {row['filename']}: {row['confidence']:.4f}\n")

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Bar chart
            ax1.bar(["Real", "Fake"], [real_count, fake_count], color=["green", "red"])
            ax1.set_title("Prediction Summary")
            ax1.set_ylabel("Number of Images")

            # Confidence histogram
            ax2.hist(confidence_scores, bins=10, color="blue", alpha=0.7)
            ax2.set_title("Confidence Distribution")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Number of Images")

            plt.tight_layout()
            plt.savefig("summary_plot.png")
            plt.show()

        else:
            print(f"{Fore.YELLOW}No valid images were processed.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error during prediction: {e}{Style.RESET_ALL}")


# --- MAIN ---
if __name__ == "__main__":

    print(f"{Fore.CYAN}=== Deepfake Detection System ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}1. Train new model{Style.RESET_ALL}")
    print(f"{Fore.CYAN}2. Predict with existing model{Style.RESET_ALL}")
    print(f"{Fore.CYAN}3. Train and predict{Style.RESET_ALL}")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        predict_unlabeled()
    elif choice == "3":
        train_model()
        predict_unlabeled()
    else:
        print(f"{Fore.RED}Invalid choice. Please run again.{Style.RESET_ALL}")