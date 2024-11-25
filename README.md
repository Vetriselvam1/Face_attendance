# Face and ID Card Recognition System
# Description
This project uses deep learning models to recognize faces and ID cards. It records attendance by matching face recognition results with ID card data and logs the results in an Excel sheet.

# Requirements
Python 3.7 or higher
Operating System: Windows (tested), Linux, or macOS
Webcam for capturing face and ID card images

# Clone the Repository
git clone https://github.com/your-username/your-repository.git
cd your-repository

# Install the necessary Python libraries using pip:
pip install opencv-python
pip install numpy
pip install openpyxl
pip install tensorflow

# Install Tkinter (if not pre-installed)
For Windows: Tkinter is included in Python.

For Linux: sudo apt-get install python3-tk

Folder Structure
Ensure your files are placed in the correct structure:

# project-folder/
│
├── attendance.xlsx               # Excel file to log attendance
├── id.h5                         # Pretrained model from techable mechine
├── person.h5                     # Pretrained model from teachhable machine
├── id_label.txt                  # Labels for ID card recognition
├── person_labels.txt             # Labels for face recognition
├── script.py                     # Python script (main program)
# How to Run
Place all required files (.h5, .txt, and attendance.xlsx) in the project folder.
Run the script using:
python script.py
# Follow the instructions on the GUI to capture images and mark attendance.
# Notes
Make sure the models (.h5 files) are trained and compatible with TensorFlow 2.x.
Ensure that your attendance.xlsx file has appropriate write permissions.
Labels in person_labels.txt and id_label.txt must match the format used in the models.
Troubleshooting
Issue: Camera not detected
Solution: Ensure your webcam is properly connected and accessible by the system.

Issue: TensorFlow version errors
Solution: Ensure TensorFlow version matches your Python version by running:

pip install tensorflow==2.5
Future Enhancements
Add support for more types of ID verification.
Implement cloud storage for attendance data.
