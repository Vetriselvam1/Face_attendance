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

# Predefined information (for all persons, including you as 'vetri')
people_info = {
    "vijay": {"id": "V001", "dob": "01/01/1990", "id_class": "vijay_id"},
    "arun": {"id": "A001", "dob": "02/02/1992", "id_class": "arun_id"},
    "priya": {"id": "P001", "dob": "03/03/1994", "id_class": "priya_id"},
    "suresh": {"id": "S001", "dob": "04/04/1996", "id_class": "suresh_id"},
    "vetri": {"id": "V002", "dob": "01/01/1995", "id_class": "vetri_id"}  # Your info added here
}

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

# Function to update Excel sheet with attendance info
def update_excel(name, id_card, dob, status, confidence):
    # Add the data into the Excel sheet (assuming columns: ID, Name, Date of Birth, Status, Confidence)
    status_message = f"{name} is {status}"  # Format the status message like "Vetri is present"
    ws.append([id_card, name, dob, status_message, confidence])
    wb.save(excel_file)

# Capture left face image
def capture_left_face():
    cap = cv2.VideoCapture(0)  # Use the default camera for face capture
    if not cap.isOpened():
        print(f"Error: Camera not found!")
        return None  # Return None if the camera is not opened

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from camera.")
            break

        cv2.putText(frame, "Please show your face and press 'c' to capture.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Face", frame)

        # Wait for user to press 'c' to capture image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            captured_image = frame
            cv2.imshow("Captured Face", captured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            return captured_image

    cap.release()
    return None  # Return None if no valid image is captured

# Capture ID card image
def capture_id_card():
    cap = cv2.VideoCapture(0)  # Use the default camera for ID card capture
    if not cap.isOpened():
        print(f"Error: Camera not found!")
        return None  # Return None if the camera is not opened

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from camera.")
            break

        cv2.putText(frame, "Please show your ID card and press 'c' to capture.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture ID Card", frame)

        # Wait for user to press 'c' to capture image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            captured_image = frame
            cv2.imshow("Captured ID Card", captured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            return captured_image

    cap.release()
    return None  # Return None if no valid image is captured

# Handle face and ID card prediction and update
def handle_prediction_and_update():
    # Step 1: Capture the left face image
    print("Capturing Left Face Image...")
    face_image = capture_left_face()
    if face_image is None:
        print("Error: Failed to capture face image.")
        return

    # Step 2: Capture the ID card image
    print("Capturing ID Card Image...")
    id_card_image = capture_id_card()
    if id_card_image is None:
        print("Error: Failed to capture ID card image.")
        return

    # Step 3: Predict classes for face and ID card
    class1, confidence1 = predict_class(face_image, model1, class_names1)
    class2, confidence2 = predict_class(id_card_image, model2, class_names2)

    # Normalize class names for consistent comparison
    class1 = class1.strip().lower()  # Face name
    class2 = class2.strip().lower()  # ID card name

    # Step 4: Show predictions in the terminal and on screen
    print(f"Face Model Prediction: Class - {class1} (Confidence: {confidence1:.2f})")
    print(f"ID Card Model Prediction: Class - {class2} (Confidence: {confidence2:.2f})")

    # Step 5: Match face with ID card for multiple persons
    if class1 == "vijay" and class2 == "vijay_id":
        print(f"Match found! Marking Vijay as Present.")
        person_info = people_info.get("vijay")
        update_excel(person_info["id"], person_info["id"], person_info["dob"], "present", confidence1)
        messagebox.showinfo("Result", "Vijay is Present")

    elif class1 == "arun" and class2 == "arun_id":
        print(f"Match found! Marking Arun as Present.")
        person_info = people_info.get("arun")
        update_excel(person_info["id"], person_info["id"], person_info["dob"], "present", confidence1)
        messagebox.showinfo("Result", "Arun is Present")

    elif class1 == "priya" and class2 == "priya_id":
        print(f"Match found! Marking Priya as Present.")
        person_info = people_info.get("priya")
        update_excel(person_info["id"], person_info["id"], person_info["dob"], "present", confidence1)
        messagebox.showinfo("Result", "Priya is Present")

    elif class1 == "suresh" and class2 == "suresh_id":
        print(f"Match found! Marking Suresh as Present.")
        person_info = people_info.get("suresh")
        update_excel(person_info["id"], person_info["id"], person_info["dob"], "present", confidence1)
        messagebox.showinfo("Result", "Suresh is Present")

    elif class1 == "vetri" and class2 == "vetri_id":
        print(f"Match found! Marking Vetri as Present.")
        person_info = people_info.get("vetri")
        update_excel(person_info["id"], person_info["id"], person_info["dob"], "present", confidence1)
        messagebox.showinfo("Result", "Vetri is Present")

    else:
        print(f"No match! Face predicted: {class1}, ID card predicted: {class2}")
        print(f"Marking as Absent.")
        # Mark as Absent in Excel for the person
        person_info = people_info.get(class1)
        if person_info:
            update_excel(person_info["id"], person_info["id"], person_info["dob"], "absent", confidence1)
        messagebox.showinfo("Result", "No match. Marked as Absent.")

# Set up the GUI using Tkinter
def setup_gui():
    root = tk.Tk()
    root.title("Face and ID Card Recognition System")
    root.geometry("500x300")

    # Label for status
    label = tk.Label(root, text="Press 'Start' to capture images and recognize", font=("Arial", 14))
    label.pack(pady=20)

    # Button to start the process
    start_button = tk.Button(root, text="Start", font=("Arial", 12), command=handle_prediction_and_update)
    start_button.pack(pady=20)

    # Exit Button
    exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit)
    exit_button.pack(pady=20)

    # Start the Tkinter event loop
    root.mainloop()

# Start the GUI
if __name__ == "__main__":
    setup_gui()
