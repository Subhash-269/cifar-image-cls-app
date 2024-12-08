import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from streamlit_js_eval import streamlit_js_eval

# Database setup
def init_db():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    # Create the table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_url TEXT,
            predicted_class TEXT,
            correct_class TEXT
        )
    ''')
    conn.commit()
    conn.close()
def add_timestamp_to_filename(filename):
    # Extract the file name and extension
    name, ext = os.path.splitext(filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{timestamp}{ext}"

def insert_image(image_url, predicted_class, correct_class=None):
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('INSERT INTO images (image_url, predicted_class, correct_class) VALUES (?, ?, ?)', (image_url, predicted_class, correct_class))
    conn.commit()
    conn.close()

# Function to delete an image entry from the database
def delete_image_from_db(image_url):
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('DELETE FROM images WHERE image_url = ?', (image_url,))
    conn.commit()
    conn.close()

# Notification placeholder at the top
notification_placeholder = st.empty()

def get_image_count():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM images')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_class_distribution():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    # Use COALESCE to fall back to predicted_class if correct_class is NULL
    c.execute('''
        SELECT COALESCE(correct_class, predicted_class) AS class, COUNT(*)
        FROM images
        GROUP BY class
    ''')
    distribution = c.fetchall()
    conn.close()
    return distribution

def get_images():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('SELECT image_url, predicted_class, correct_class FROM images')
    images = c.fetchall()
    conn.close()
    return images

def clear_db():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('DELETE FROM images')
    conn.commit()
    conn.close()

# Function to show class distribution
def show_class_distribution():
    class_distribution = get_class_distribution()
    if class_distribution:
        classes, counts = zip(*class_distribution)
        classes = list(map(str, classes))  # Ensure classes are strings
        counts = list(map(int, counts))   # Ensure counts are integers

        fig, ax = plt.subplots()
        ax.bar(classes, counts, color='skyblue')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.write("No class distribution data available.")


# Function to calculate and show accuracy
def show_accuracy():
    images = get_images()
    if images:
        # Count correct predictions where correct_class matches predicted_class
        correct_count = sum(1 for _, predicted, correct in images if correct == predicted)
        incorrect_count = len(images) - correct_count

        labels = ['Correct', 'Incorrect']
        counts = [correct_count, incorrect_count]
        
        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        ax.set_title('Prediction Accuracy')
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.write("No accuracy data available.")


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Load the state dict and handle key mismatches
state_dict = torch.load("best_model.pth", map_location=device)
new_state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict the class of an image
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


# Function to save and clear the home screen
def final_save(image_path, predicted_class, correct_class=None):
    insert_image(image_path, predicted_class, correct_class)  # Save to database
    st.session_state["image_history"] = get_images()  # Refresh session state with updated history
    st.success("Image and class saved to the database.")  # Notify the user
    st.experimental_rerun()  # Clear the screen by re-running the app


# Streamlit app
st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it or provide a URL to an image.")

# Initialize the database
init_db()  # Only initialize the database, do not clear it
# clear_db()

# Dashboard to show the number of images uploaded
image_count = get_image_count()
# st.sidebar.header("Dashboard")
# st.sidebar.write(f"Total images uploaded: {image_count}")

# Get class distribution for visualization
# class_distribution = get_class_distribution()


# Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# Multi-file upload
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Input for image URL
image_url = st.text_input("Or enter the URL of an image:")

image = None
predicted_class = None

# if uploaded_file is not None:
#     # Save the uploaded image to a temporary location
#     image = Image.open(uploaded_file)
#     temp_file_path = f"temp_images/{uploaded_file.name}"
#     os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
#     image.save(temp_file_path)
#     image_url = temp_file_path  # Use the path for saving in the database
# elif image_url:
#     try:
#         response = requests.get(image_url)
#         image = Image.open(BytesIO(response.content))
#     except Exception as e:
#         st.error(f"Error loading image from URL: {e}")
#         image = None

import time

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Save each uploaded image to a temporary location
            image = Image.open(uploaded_file)
            
            # Add timestamp to ensure unique file path
            timestamp = int(time.time())
            temp_file_path = f"temp_images/{timestamp}_{uploaded_file.name}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            image.save(temp_file_path)

            # Check for duplicate image based on file name
            existing_images = get_images()
            duplicate_found = any(uploaded_file.name in image_url for image_url, _, _ in existing_images)

            if duplicate_found:
                st.warning(f"The image '{uploaded_file.name}' already exists.")
                replace = st.radio(
                    f"Would you like to replace '{uploaded_file.name}'?",
                    options=["Replace", "Ignore"],
                    key=f"replace_{uploaded_file.name}"
                )

                if replace == "Replace":
                    # Remove the old entry and save the new one
                    delete_image_from_db(f"%{uploaded_file.name}")
                    st.success(f"Replaced image '{uploaded_file.name}'.")
                elif replace == "Ignore":
                    st.info(f"Ignored duplicate image '{uploaded_file.name}'.")
                    continue  # Skip further processing for this image

            # Predict the class
            st.write(f"Processing {uploaded_file.name}...")
            class_id = predict(image)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_class = class_names[class_id]

            # Side-by-side layout
            col1, col2 = st.columns([1, 2])  # Adjust column widths

            with col1:
                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

            with col2:
                st.write(f"**Predicted class:** {predicted_class}")
                st.write(f"Is the predicted class correct for {uploaded_file.name}?")
                correct_class = st.radio("", options=["Yes", "No"], key=f"correct_{uploaded_file.name}")

                if correct_class == "No":
                    correct_class_name = st.selectbox(
                        f"Select the correct class for {uploaded_file.name}:",
                        class_names,
                        key=f"correct_class_{uploaded_file.name}"
                    )
                else:
                    correct_class_name = predicted_class

                if st.button(f"Save {uploaded_file.name}", key=f"save_button_{uploaded_file.name}"):
                    insert_image(temp_file_path, predicted_class, correct_class_name)
                    st.success(f"Image {uploaded_file.name} and its class saved to the database.")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # Dynamically update session state with the new entries
    st.session_state["image_history"] = get_images()


# if image is not None:
#     st.image(image, caption='Uploaded or Fetched Image', use_container_width=True)
#     st.write("Classifying...")

#     # Predict the class
#     class_id = predict(image)
#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     predicted_class = class_names[class_id]
#     st.write(f"Predicted class: {predicted_class}")

#     # Feedback section
#     correct_class = st.selectbox("Is the predicted class correct?", options=["Yes", "No"])
#     if correct_class == "No":
#         correct_class_name = st.selectbox("Select the correct class:", class_names)
#     else:
#         correct_class_name = None

    # Save button
    # if st.button("Save"):
    #     if image_url:
    #         insert_image(image_url, predicted_class, correct_class_name)
    #     else:
    #         insert_image(temp_file_path, predicted_class, correct_class_name)  # Save the path of the uploaded image
    #     st.success("Image and class saved to the database.")
    #     st.stop()  # Stop execution to refresh the app

# Display the dashboard on the left sidebar


# Show history of images and predicted classes dynamically
def show_history_on_dashboard():
    if "image_history" not in st.session_state:
        # Load initial data
        st.session_state["image_history"] = get_images()

    images = st.session_state["image_history"]
    if images:
        st.sidebar.subheader("Image History")
        
        # Display each image and its associated details
        for image_url, predicted_class, correct_class in images:
            try:
                # Display the image
                if os.path.exists(image_url):
                    image = Image.open(image_url)
                else:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                
                # Show the image and class details
                display_correct_class = correct_class if correct_class else predicted_class
                st.sidebar.image(image, caption=f"Predicted: {predicted_class} | Corrected: {display_correct_class}", use_container_width=True)
                st.sidebar.write(f"**Predicted Class:** {predicted_class}")
                st.sidebar.write(f"**Corrected Class:** {display_correct_class}")
                st.sidebar.markdown("---")
            except Exception as e:
                st.sidebar.error(f"Error loading image {image_url}: {e}")
    else:
        st.sidebar.write("No saved images to display.")

# Show history of images and predicted classes dynamically in a compact layout for the dashboard
def show_history_compact():
    if "image_history" not in st.session_state:
        # Load initial data
        st.session_state["image_history"] = get_images()

    images = st.session_state["image_history"]
    if images:
        st.sidebar.subheader("Image History")
        
        # Display images and their details in a compact table-like view
        for image_url, predicted_class, correct_class in images:
            try:
                # Display the image with reduced size
                if os.path.exists(image_url):  # Local file
                    image = Image.open(image_url)
                else:  # URL
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))

                # Create a row layout in the sidebar
                col1, col2, col3, col4 = st.sidebar.columns([1, 2, 2, 1])  # Adjust column width ratios
                
                with col1:
                    st.image(image, use_container_width=True, caption=None)
                
                with col2:
                    st.write(f"**Predicted:** {predicted_class}")
                
                with col3:
                    corrected_text = correct_class if correct_class else predicted_class
                    st.write(f"**Corrected:** {corrected_text}")

                with col4:
                    # Add a delete button with a red X emoji
                    delete_button = st.button("❌", key=f"delete_{image_url}")
                    if delete_button:
                        delete_image_from_db(image_url)
                        # Display notification at the top left
                        notification_placeholder.success(f"Deleted entry for {image_url}")
                        # Refresh the session state and dashboard
                        st.session_state["image_history"] = get_images()
                        # st.experimental_rerun()  # Refresh the app to reflect changes
                
                st.sidebar.markdown("---")  # Separator for rows
            except Exception as e:
                st.sidebar.error(f"Error loading image {image_url}: {e}")
    else:
        st.sidebar.write("No saved images to display.")

# Display the history in the sidebar
# show_history_on_dashboard()

# show_history_compact()

def show_dashboard():
    image_count = get_image_count()
    st.sidebar.header("Dashboard")
    st.sidebar.write(f"Total images uploaded: {image_count}")

    # Add a Clear Database button in the dashboard
    if st.sidebar.button("Clear Database"):
        clear_db()
        st.success("Database cleared.")
        st.session_state["image_history"] = []  # Clear the session state
        # st.experimental_rerun()  # Refresh the app
    
    # Add visualizations
    st.sidebar.subheader("Visualizations")
    show_class_distribution()  # Show Class vs Count
    show_accuracy()            # Show Accuracy

    # Show history in a compact format
    show_history_compact()


# Include the dashboard in the sidebar
show_dashboard()

# Save the image and dynamically update the dashboard


if st.button("Clear All"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


# Include the history in the dashboard
# show_history_on_dashboard()


