import cv2
import numpy as np
import openpyxl
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox

# Load Keras models for face and ID card
model1 = load_model('C:/Users/sugas/Desktop/New folder/person.h5', compile=False)  # Model for face recognition
model2 = load_model('C:/Users/sugas/Desktop/New folder/id.h5', compile=False)  # Model for ID card recognition

# Load labels for both models and clean up class names
def load_class_names(filename):
    with open(filename, "r") as file:
        class_names = file.readlines()
    return [class_name.strip().split()[-1] for class_name in class_names]  # Extract the last part of each label, e.g., "Class 1"

# Load class names for each model
class_names1 = load_class_names("C:/Users/sugas/Desktop/New folder/person_labels.txt")  # Labels for face model
class_names2 = load_class_names("C:/Users/sugas/Desktop/New folder/id_label.txt")  # Labels for ID card model

# Excel setup
excel_file = 'C:/Users/sugas/Desktop/New folder/attendance.xlsx'
wb = openpyxl.load_workbook(excel_file)
ws = wb.active

# Predefined information
name = "Vetri"
id_card = "RBM056"
dob = "01/01/1990"  # Date of birth can be added if necessary

# Function to preprocess image and predict class
def predict_class(image, model, class_names):
    if image is None:
        print("Error: No image captured!")
        return None, 0  # Return default values if image is None
    
    # Preprocess the image as per model requirements
    image_resized = cv2.resize(image, (224, 224))  # Adjust size if needed
    image_array = np.asarray(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize

    # Get model prediction
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]  # Get cleaned class name
    confidence = prediction[0][index]
    return class_name, confidence

# Capture image and display prediction in real-time
def capture_and_predict(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera not found!")
        return None  # Return None if the camera is not opened

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from camera.")
            break

        # Predict class for face and ID card in real-time
        class1, confidence1 = predict_class(frame, model1, class_names1)
        class2, confidence2 = predict_class(frame, model2, class_names2)

        # Normalize class names for consistent comparison
        class1 = class1.strip().lower()
        class2 = class2.strip().lower()

        # Show the predictions on the frame
        cv2.putText(frame, f"Face: {class1} (Conf: {confidence1:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {class2} (Conf: {confidence2:.2f})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print the predictions to the terminal
        print(f"Face Model Prediction: Class - {class1} (Confidence: {confidence1:.2f})")
        print(f"ID Card Model Prediction: Class - {class2} (Confidence: {confidence2:.2f})")

        # Display the updated frame with predictions
        cv2.imshow("Camera Feed with Predictions", frame)

        # Press 'c' to capture image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print(f"Captured: Face: {class1}, ID: {class2}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Set up the GUI using Tkinter
def setup_gui():
    root = tk.Tk()
    root.title("Face and ID Card Recognition System")
    root.geometry("500x300")

    # Label for status
    label = tk.Label(root, text="Press 'Start' to capture images and recognize", font=("Arial", 14))
    label.pack(pady=20)

    # Button to start the process
    start_button = tk.Button(root, text="Start", font=("Arial", 12), command=lambda: capture_and_predict(0))
    start_button.pack(pady=20)

    # Exit Button
    exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit)
    exit_button.pack(pady=20)

    # Start the Tkinter event loop
    root.mainloop()

# Start the GUI
if __name__ == "__main__":
    setup_gui()
