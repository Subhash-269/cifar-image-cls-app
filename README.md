# CIFAR-10 Image Classification App

![Raw_App_Screenshot](https://github.com/user-attachments/assets/c3b57aff-c6a6-4fa3-9312-74927a61d59a)


## **Overview**
The CIFAR-10 Image Classification Service is a machine learning application designed to classify images into one of ten categories using a ResNet-18 model. The app is built with Streamlit and supports uploading multiple image files or providing image URLs for classification. It includes features for saving predictions, correcting predictions, visualizing classification data, and downloading data for further analysis or retraining.

---

## **Features**
- **Image Classification**:
  - Classifies images into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
  - Allows users to provide custom labels for images outside the predefined categories.

- **File Support**:
  - Accepts images via file upload.

- **Feedback System**:
  - Allows users to validate or correct predictions with flexibility.
  - <img src="https://github.com/user-attachments/assets/4cffcad1-0921-47bc-938f-c5d1d66a9920" width="50%">


- **Data Visualization**:
  - Displays the class distribution (correct vs. incorrect predictions) using horizontal stacked bar charts.
  - Visualizes overall prediction accuracy.
  - <img src="https://github.com/user-attachments/assets/9eb26111-822b-4449-9e19-cccd97e96666" width = "50%">


- **Database Integration**:
  - Stores images and their classifications in an SQLite database.
  - Allows users to view and filter image history (all, correct, or incorrect predictions).
  - <img src="https://github.com/user-attachments/assets/0af54f27-93f0-4bc6-8182-08c59ecaf701" width="50%">


- **Data Management**:
  - Clear individual or all database entries.
  - Multi-select file upload.
  - Download images and their corrected labels as ZIP files for training or analysis.

- **Error Handling**:
  - Automatically filters invalid input types and gracefully handles errors.

---

## **Usage Instructions**

### **1. Running the App Locally**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/cifar10-image-classification.git
   cd cifar10-image-classification
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Launch the Streamlit app:
   ```bash
   streamlit run app_.py
   ```

4. **Access the App**:
   Open the URL provided in the terminal (default: `http://localhost:8501`).

---

### **2. Using the App**
1. **Upload Images**:
   - Use the file uploader to select one or more images (JPEG or PNG format).
   - Alternatively, provide a direct URL to an image.

2. **View Predictions**:
   - The app predicts the class of each uploaded image and displays it with a confidence score.
   - If the prediction is incorrect, users can correct it by selecting a label from the predefined categories or providing a custom label.

3. **Save Data**:
   - Save the predictions and corrections to the database by clicking the "Save" button.

4. **Visualize Data**:
   - View the class distribution and prediction accuracy in the sidebar.

5. **Manage Data**:
   - Use the "Clear Database" button to delete all saved data.
   - Use the "Clear All" button to refresh the app interface.

6. **Download Data**:
   - Download all images and their corrected labels, or filter specific datasets (e.g., incorrect predictions).

---

## **Project Structure**
- `app.py`: Main application file.
- `requirements.txt`: List of required dependencies.
- `image_classification.db`: SQLite database for storing image data (auto-created).
- `temp_images/`: Temporary storage for uploaded images.

---

## **Dependencies**
The project uses the following libraries and frameworks:
- `streamlit`
- `torch`
- `torchvision`
- `Pillow`
- `requests`
- `sqlite3`
- `plotly`
- `st_aggrid`
- `streamlit-js-eval`
- `zipfile`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## **Model and Dataset**

### **Model**
The application uses a fine-tuned **ResNet-18** model, optimized for classifying CIFAR-10 images into 10 predefined categories. The model architecture is robust and efficient, leveraging residual connections for improved performance.

### **Dataset**
The **CIFAR-10 dataset** consists of 60,000 color images (32x32 pixels) categorized into the following classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

**Dataset Details:**
- **Training Data:** 60,000 images.
- **Test Data:** 10,000 images.
- **Image Format:** RGB color images with equal representation across categories.

---

## **Acknowledgments**
- The CIFAR-10 dataset.
- ResNet architecture by Kaiming He et al.
