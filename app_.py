import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO
import sqlite3
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import os
import datetime
from streamlit_js_eval import streamlit_js_eval
import base64
import zipfile
import tempfile


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
    
    # Add the `confidence_score` column if it doesn't exist
    try:
        c.execute("ALTER TABLE images ADD COLUMN confidence_score REAL")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    conn.commit()
    conn.close()


def add_timestamp_to_filename(filename):
    # Extract the file name and extension
    name, ext = os.path.splitext(filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{timestamp}{ext}"

def insert_image(image_url, predicted_class, correct_class=None, confidence_score=None):
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO images (image_url, predicted_class, correct_class, confidence_score) VALUES (?, ?, ?, ?)',
        (image_url, predicted_class, correct_class, confidence_score)
    )
    conn.commit()
    conn.close()
    
    # Update session state after saving
    if "image_history" in st.session_state:
        st.session_state["image_history"] = get_images()  # Refresh session data



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
    c.execute('SELECT image_url, predicted_class, correct_class, confidence_score FROM images')
    images = c.fetchall()
    conn.close()
    return images



def clear_db():
    conn = sqlite3.connect('image_classification.db')
    c = conn.cursor()
    c.execute('DELETE FROM images')
    conn.commit()
    conn.close()

def create_thumbnail(image_path):
    """Create a thumbnail for the given image path and return it as a Base64 HTML string."""
    with Image.open(image_path) as img:
        img.thumbnail((60, 60))  # Create a small thumbnail
        buffer = BytesIO()
        img.save(buffer, format="JPEG")  # Save thumbnail to a buffer
        buffer.seek(0)
        encoded_thumbnail = base64.b64encode(buffer.read()).decode("utf-8")
    return f'<img src="data:image/jpeg;base64,{encoded_thumbnail}" style="height:60px; width:60px;">'


# Function to show class distribution
def show_class_distribution():
    # Fetch all images from the database
    images = get_images()
    if images:
        # Extract unique classes
        class_set = set(image[1] for image in images).union(
            image[2] for image in images if image[2] is not None
        )
        classes = sorted(list(class_set))  # Sort classes alphabetically
        
        # Initialize correct and incorrect counts for all classes
        correct_counts = {cls: 0 for cls in classes}
        incorrect_counts = {cls: 0 for cls in classes}

        # Iterate through images and calculate counts
        for _, predicted_class, correct_class, _ in images:
            if correct_class == predicted_class:
                correct_counts[correct_class] += 1
            else:
                # Increment incorrect count for the actual class
                incorrect_counts[correct_class] += 1

        # Prepare data for the stacked bar chart
        correct_values = [correct_counts[cls] for cls in classes]
        incorrect_values = [incorrect_counts[cls] for cls in classes]

        # Create the stacked horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=correct_values,
            y=classes,
            orientation="h",
            name="Correct Predictions",
            marker=dict(color="green")
        ))
        fig.add_trace(go.Bar(
            x=incorrect_values,
            y=classes,
            orientation="h",
            name="Incorrect Predictions",
            marker=dict(color="red")
        ))

        # Update layout
        fig.update_layout(
            title="Class Distribution (Correct vs Incorrect Predictions)",
            xaxis_title="Count",
            yaxis_title="Classes",
            barmode="stack",  # Enable stacking of bars
            xaxis=dict(
                tickmode="linear",
                dtick=1,  # Ensure tick marks are integers
            ),
        )

        # Display the chart in the sidebar
        st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.write("No class distribution data available.")


# Function to calculate and show accuracy
def show_accuracy():
    images = get_images()
    if images:
        # Count correct predictions where correct_class matches predicted_class
        correct_count = sum(1 for _, predicted, correct, _ in images if correct == predicted)
        incorrect_count = len(images) - correct_count

        labels = ['Correct', 'Incorrect']
        counts = [correct_count, incorrect_count]

        # Create a pie chart with Plotly
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.3,
                marker=dict(colors=["green", "red"]),
            )
        )
        fig.update_layout(
            title="Prediction Accuracy",
        )

        st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.write("No accuracy data available.")


temp_dirs = {}  # Store references to TemporaryDirectory to avoid cleanup

def prepare_images_for_download(filter_type):
    """Prepare images and labels for downloading based on the selected filter."""
    global temp_dirs  # To store temp directories and avoid early cleanup

    # Retrieve filtered images
    images = []
    if filter_type == "all":
        images = get_images()
    elif filter_type == "incorrect":
        images = [
            (image_url, predicted_class, correct_class, confidence_score)
            for image_url, predicted_class, correct_class, confidence_score in get_images()
            if correct_class is not None and predicted_class != correct_class
        ]
    elif filter_type == "correct":
        images = [
            (image_url, predicted_class, correct_class, confidence_score)
            for image_url, predicted_class, correct_class, confidence_score in get_images()
            if predicted_class == correct_class
        ]

    if not images:
        st.warning("No images available for the selected filter.")
        return None

    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_dirs[filter_type] = temp_dir  # Keep the reference to avoid cleanup
    zip_path = os.path.join(temp_dir.name, f"{filter_type}_images.zip")

    # Prepare labels.txt file
    labels_file_path = os.path.join(temp_dir.name, "labels.txt")
    with open(labels_file_path, "w") as labels_file:
        labels_file.write("Image_Name,Corrected_Label\n")  # Add header

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for image_url, predicted_class, correct_class, confidence_score in images:
                try:
                    # Load the image
                    if os.path.exists(image_url):
                        image = Image.open(image_url)
                    else:
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))

                    # Use corrected label if available; otherwise, use predicted class
                    label = correct_class if correct_class else predicted_class
                    image_name = f"{label}_{os.path.basename(image_url)}"
                    image_path = os.path.join(temp_dir.name, image_name)

                    # Save the image temporarily
                    image.save(image_path)

                    # Write image name and label to labels.txt
                    labels_file.write(f"{image_name},{label}\n")

                    # Add the image to the ZIP archive
                    zipf.write(image_path, arcname=image_name)

                except Exception as e:
                    st.error(f"Error processing image {image_url}: {e}")

            # Add the labels file to the ZIP archive
            labels_file.flush()  # Ensure all content is written to the file
            os.fsync(labels_file.fileno())  # Force the OS to write to disk
            zipf.write(labels_file_path, arcname="labels.txt")

    # Debug: Check if labels.txt is populated correctly
    # with open(labels_file_path, "r") as debug_file:
    #     debug_content = debug_file.read()
    #     st.text(f"Labels content:\n{debug_content}")  # This will display in Streamlit

    return zip_path





def download_images_button(filter_type, label):
    """Create a button to download images and labels as a ZIP file."""
    zip_path = prepare_images_for_download(filter_type)
    if zip_path:
        with open(zip_path, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=f"{filter_type}_images.zip",
                mime="application/zip",
            )



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
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()



# Function to save and clear the home screen
def final_save(image_path, predicted_class, correct_class=None):
    insert_image(image_path, predicted_class, correct_class)  # Save to database
    st.session_state["image_history"] = get_images()  # Refresh session state with updated history
    st.success("Image and class saved to the database.")  # Notify the user
    st.experimental_rerun()  # Clear the screen by re-running the app


# Streamlit app
st.markdown(
    """
    <h1 style="text-align: center; font-size: 2.5rem;">EAI 6010:Module 5: Assignment - Microservice for a model</h1>
    <h2 style="text-align: center; font-size: 1.8rem; color: #555;">CIFAR-10 Image Classification</h2>
    """,
    unsafe_allow_html=True
)
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
            # Save the uploaded image
            image = Image.open(uploaded_file)
            timestamp = int(time.time())
            temp_file_path = f"temp_images/{timestamp}_{uploaded_file.name}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            image.save(temp_file_path)

            # Predict class and confidence
            st.write(f"Processing {uploaded_file.name}...")
            class_id, confidence = predict(image)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_class = class_names[class_id]

            # Display results
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            with col2:
                st.write(f"**Predicted class:** {predicted_class}")
                st.write(f"**Confidence score:** {confidence:.2f}")
                correct_class = st.radio(f"Is the prediction correct for {uploaded_file.name}?", ["Yes", "No"])
                if correct_class == "No":
                    correct_class_name = st.selectbox("Select the correct class:", class_names)
                else:
                    correct_class_name = predicted_class

                if st.button(f"Save {uploaded_file.name}", key=f"save_button_{uploaded_file.name}"):
                    insert_image(temp_file_path, predicted_class, correct_class_name, confidence)
                    st.success(f"Image '{uploaded_file.name}' and its class saved to the database.")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


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
        st.sidebar.subheader("All Images")
        
        # Display each image and its associated details
        for image_url, predicted_class, correct_class, confidence_score in images:
            try:
                # Display the image
                if os.path.exists(image_url):
                    image = Image.open(image_url)
                else:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                
                # Display image and details
                st.sidebar.image(image, caption=f"Predicted: {predicted_class} | Confidence: {confidence_score:.2f}", use_container_width=True)
                corrected_text = correct_class if correct_class else predicted_class
                st.sidebar.write(f"**Corrected Class:** {corrected_text}")
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
        st.sidebar.subheader("All Images")
        
        # Display images and their details in a compact table-like view
        for image_url, predicted_class, correct_class, confidence_score in images:
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
                    st.write(f"**Confidence:** {confidence_score:.2f}")

                with col4:
                    # Add a delete button with a red X emoji
                    delete_button = st.button("❌", key=f"delete_{image_url}")
                    if delete_button:
                        delete_image_from_db(image_url)
                        # Display notification at the top left
                        notification_placeholder.success(f"Deleted entry for {image_url}")
                        # Refresh the session state and dashboard
                        st.session_state["image_history"] = get_images()
                
                st.sidebar.markdown("---")  # Separator for rows
            except Exception as e:
                st.sidebar.error(f"Error loading image {image_url}: {e}")
    else:
        st.sidebar.write("No saved images to display.")

if st.button("Clear All"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def show_history_in_main_area():
    st.subheader("All Images")
    
    images = get_images()
    if images:
        for image_url, predicted_class, correct_class, confidence_score in images:
            try:
                # Display the image
                if os.path.exists(image_url):
                    image = Image.open(image_url)
                else:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))

                # Create layout for each entry
                col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
                
                with col1:
                    st.image(image, use_container_width=True)

                with col2:
                    st.write(f"**Predicted:** {predicted_class}")
                    corrected_text = correct_class if correct_class else predicted_class
                    st.write(f"**Corrected:** {corrected_text}")
                    st.write(f"**Confidence:** {confidence_score:.2f}")

                with col3:
                    delete_button = st.button("❌", key=f"delete_{image_url}")
                    if delete_button:
                        # Show confirmation dialog dynamically
                        st.write("**Are you sure you want to delete this entry?**")
                        confirm = st.radio(
                            "",
                            options=["Yes", "No"],
                            key=f"confirm_delete_{image_url}",
                            horizontal=True,
                        )

                        if confirm == "Yes":
                            delete_image_from_db(image_url)
                            st.success(f"Deleted entry for {image_url}.")
                            st.experimental_rerun()
                        elif confirm == "No":
                            st.info("Deletion cancelled.")
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading image {image_url}: {e}")
    else:
        st.write("No saved images to display.")


def show_image_filters():
    """Main function to display filters and respective image views."""
    st.title("Database")
    
    # Create the buttons in a single row
    selected_view = st.radio(
        "Select View:",
        options=["All Images", "Incorrect Predictions", "Correct Predictions"],
        horizontal=True,
    )

    # Dynamically load the selected view
    if selected_view == "All Images":
        st.subheader("All Images")
        all_images = filter_images("all")
        display_images(all_images)

    elif selected_view == "Incorrect Predictions":
        st.subheader("Incorrect Predictions")
        incorrect_images = filter_images("incorrect")
        display_images(incorrect_images)

    elif selected_view == "Correct Predictions":
        st.subheader("Correct Predictions")
        correct_images = filter_images("correct")
        display_images(correct_images)


def filter_images(filter_type):
    """Filter images based on the filter type."""
    images = get_images()
    if filter_type == "incorrect":
        return [
            (image_url, predicted_class, correct_class, confidence_score)
            for image_url, predicted_class, correct_class, confidence_score in images
            if predicted_class != correct_class
        ]
    elif filter_type == "correct":
        return [
            (image_url, predicted_class, correct_class, confidence_score)
            for image_url, predicted_class, correct_class, confidence_score in images
            if predicted_class == correct_class
        ]
    return images  # Return all images for "history"


def display_images(filtered_images):
    """Display images in the specified format."""
    if filtered_images:
        for image_url, predicted_class, correct_class, confidence_score in filtered_images:
            try:
                # Load image
                if os.path.exists(image_url):
                    image = Image.open(image_url)
                else:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))

                # Create layout for each entry
                col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
                
                with col1:
                    st.image(image, use_container_width=True)

                with col2:
                    st.write(f"**Predicted:** {predicted_class}")
                    corrected_text = correct_class if correct_class else predicted_class
                    st.write(f"**Corrected:** {corrected_text}")
                    st.write(f"**Confidence:** {confidence_score:.2f}")

                with col3:
                    delete_button = st.button("❌", key=f"delete_{image_url}")
                    if delete_button:
                        # Show confirmation dialog dynamically
                        st.write("**Are you sure you want to delete this entry?**")
                        confirm = st.radio(
                            "",
                            options=["Yes", "No"],
                            key=f"confirm_delete_{image_url}",
                            horizontal=True,
                        )

                        if confirm == "Yes":
                            delete_image_from_db(image_url)
                            st.success(f"Deleted entry for {image_url}.")
                            st.experimental_rerun()
                        elif confirm == "No":
                            st.info("Deletion cancelled.")
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading image {image_url}: {e}")
    else:
        st.write("No saved images to display.")

def show_filterable_image_history():
    # Fetch All Images from the database
    images = get_images()
    if images:
        # Prepare data for the table
        data = [
            {
                "Image Path": image_url,
                "Predicted Class": predicted_class,
                "Correct Class": correct_class,
                "Confidence Score": round(confidence_score, 2),
            }
            for image_url, predicted_class, correct_class, confidence_score in images
        ]
        df = pd.DataFrame(data)

        # Configure the AgGrid table
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=False, filterable=True)
        gb.configure_selection("single")  # Allow single-row selection
        gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
        grid_options = gb.build()

        # Render the AgGrid table
        st.subheader("Filterable All Images")
        response = AgGrid(
            df,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            allow_unsafe_jscode=False,
            theme="streamlit",
        )

        # Display selected image
        selected_rows = response["selected_rows"]
        if selected_rows:
            selected_image_path = selected_rows[0]["Image Path"]
            if os.path.exists(selected_image_path):
                st.image(selected_image_path, caption=f"Selected Image: {selected_image_path}", use_column_width=True)
            else:
                st.error(f"Image not found: {selected_image_path}")
    else:
        st.write("No saved images to display.")

def show_dashboard():
    image_count = get_image_count()
    # st.sidebar.header("Module 5: Assignment - Microservice for a model")
    st.sidebar.header("Author: Venkat Neelraj Nitta")
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
    # Buttons for each view
    

    # Show history in a compact format
    # show_history_compact()
    # show_history_in_main_area()
    # show_filterable_image_history()
    show_image_filters()
    

# Include the dashboard in the sidebar
show_dashboard()

st.subheader("Download Data for Training")

col1, col2, col3 = st.columns(3)

with col1:
    # st.markdown("### All Images")
    download_images_button("all", "Download All Images and Labels")

with col2:
    # st.markdown("### Incorrect Predictions")
    download_images_button("incorrect", "Download Incorrect Predictions and Labels")

with col3:
    # st.markdown("### Correct Predictions")
    download_images_button("correct", "Download Correct Predictions and Labels")

# def add_footer():
#     st.markdown(
#         """
#         <hr style="margin-top: 3rem;">
#         <footer style="text-align: center; font-size: 0.9rem; color: #999;">
#             <strong>Module 5: Assignment - Microservice for a model</strong><br>
#             Author - Venkat Neelraj Nitta
#         </footer>
#         """,
#         unsafe_allow_html=True
#     )

# # Add the footer at the end of the app
# add_footer()


# Save the image and dynamically update the dashboard


# if st.button("Clear All"):
#     streamlit_js_eval(js_expressions="parent.window.location.reload()")


# Include the history in the dashboard
# show_history_on_dashboard()


