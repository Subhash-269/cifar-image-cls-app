# CIFAR-10 Image Classification Service

## **Overview**
The CIFAR-10 Image Classification Service is a machine learning application designed to classify images into one of ten categories using a ResNet-18 model. The app is built with Streamlit and supports uploading multiple image files or providing image URLs for classification. It includes features for saving predictions, correcting predictions, and visualizing classification data.

---

## **Features**
- **Image Classification**: Classifies images into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
- **File and URL Support**: Accepts images via file upload or direct URLs.
- **Feedback System**: Allows users to correct predictions if needed.
- **Data Visualization**:
  - Class distribution of uploaded images.
  - Prediction accuracy (correct vs. incorrect predictions).
- **Database Integration**: Stores images and their classifications in an SQLite database.
- **Data Management**:
  - Clear all database entries.
  - Refresh app interface.

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
   streamlit run app.py
   ```

4. **Access the App**:
   Open the URL provided in the terminal (default: `http://localhost:8501`).

---

### **2. Using the App**
1. **Upload Images**:
   - Use the file uploader to select one or more images (JPEG or PNG format).
   - Alternatively, provide a direct URL to an image.

2. **View Predictions**:
   - The app predicts the class of each uploaded image and displays it.
   - Predictions can be confirmed or corrected by the user.

3. **Save Data**:
   - Save the predictions and corrections to the database by clicking the "Save" button.

4. **Visualize Data**:
   - View the class distribution and prediction accuracy on the sidebar.

5. **Manage Data**:
   - Use the "Clear Database" button to delete all saved data.
   - Use the "Clear All" button to refresh the app interface.

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
- `sqlite3`
- `matplotlib`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- The CIFAR-10 dataset.
- ResNet architecture by Kaiming He et al.

---


