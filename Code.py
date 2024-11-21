import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Button, Canvas, font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize matplotlib
plt.style.use('ggplot')


# Function to extract features from an image
def extract_image_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open/read image file: {image_path}")
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    features = np.concatenate((hist_b, hist_g, hist_r)).flatten()
    return features


# Initialize model and scaler at global scope
scaler = None
rf_classifier = None


# Function to display an image in the GUI
def display_image_on_canvas(image_path, canvas):
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Updated resizing method
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(10, 10, anchor='nw', image=img_tk)
    canvas.image = img_tk  # Keep a reference, prevent GC
    canvas.update()


# Process images and train model
def process_images_and_train(canvas):
    global scaler, rf_classifier
    image_dir_clean = filedialog.askdirectory(title="Select directory of clean images")
    image_dir_stego = filedialog.askdirectory(title="Select directory of stego images")
    if not image_dir_clean or not image_dir_stego:
        messagebox.showwarning("Training Cancelled", "Model training cancelled. Please select both directories.")
        return

    feature_list = []
    label_list = []
    process_images(image_dir_clean, 'clean', feature_list, label_list, canvas)
    process_images(image_dir_stego, 'stego', feature_list, label_list, canvas)
    scaler, rf_classifier = setup_model(feature_list, label_list)


def process_images(image_dir, label, feature_list, label_list, canvas):
    if not os.path.exists(image_dir):
        messagebox.showerror("Directory Not Found", f"Directory {image_dir} does not exist.")
        return
    for filename in tqdm(os.listdir(image_dir), desc=f'Processing {label} images'):
        if filename.endswith(('.jpg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, filename)
            try:
                features = extract_image_features(image_path)
                feature_list.append(features)
                label_list.append(label)
                display_image_on_canvas(image_path, canvas)  # Display each image as it's processed
            except FileNotFoundError as e:
                messagebox.showerror("File Not Found", str(e))


def setup_model(feature_list, label_list):
    if not feature_list:
        messagebox.showerror("Error", "No features extracted. Check your image directories and try again.")
        return None, None

    df_features = pd.DataFrame(feature_list)
    df_labels = pd.DataFrame(label_list, columns=['label'])
    df = pd.concat([df_features, df_labels], axis=1)

    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    display_model_results(y_test, y_pred)

    return scaler, rf_classifier


def display_model_results(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    messagebox.showinfo("Model Performance", f'Accuracy: {acc}\n{classification_report(y_test, y_pred)}')

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    # Plot precision, recall, f1-score
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(columns=['support'])

    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 5))
    plt.title('Precision, Recall, and F1-Score')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.show()


def predict_image():
    global scaler, rf_classifier
    if scaler is None or rf_classifier is None:
        messagebox.showerror("Error", "The model is not trained. Please train the model first.")
        return
    image_path = filedialog.askopenfilename(title="Select an image file to test",
                                            filetypes=[("Image files", "*.jpg *.png *.bmp")])
    if image_path:
        try:
            features = extract_image_features(image_path)
        except FileNotFoundError as e:
            messagebox.showerror("File Not Found", str(e))
            return
        features_scaled = scaler.transform([features])
        prediction = rf_classifier.predict(features_scaled)
        messagebox.showinfo("Prediction Result", f"The image is predicted to be: {prediction[0]}")


# GUI Setup
def setup_gui():
    root = tk.Tk()
    root.title("Steganography Detection Tool")
    root.geometry("500x600")  # Set the size of the window

    title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
    button_font = tkfont.Font(family='Helvetica', size=12)

    canvas = Canvas(root, width=260, height=260)
    canvas.pack(pady=20)

    button_frame = Frame(root)
    button_frame.pack(fill='both', expand=True, padx=20, pady=20)

    train_button = Button(button_frame, text="Train Model", command=lambda: process_images_and_train(canvas),
                          font=button_font)
    train_button.pack(fill='x', expand=True, pady=10)

    predict_button = Button(button_frame, text="Predict on Image", command=predict_image,
                            font=button_font)
    predict_button.pack(fill='x', expand=True, pady=10)

    exit_button = Button(button_frame, text="Exit", command=root.destroy, font=button_font)
    exit_button.pack(fill='x', expand=True, pady=10)

    root.mainloop()


if __name__ == "__main__":
    setup_gui()
    root = tk
