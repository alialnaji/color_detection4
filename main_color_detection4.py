import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import joblib

# Configure the page using st.set_page_config()
st.set_page_config(
    page_title="Tongue Color Detection Streamlit App",
    page_icon="👅",     
    layout="centered",   #"wide"
    initial_sidebar_state="collapsed",        # "expanded"
)

# Load the YOLO model
#model = YOLO('C:/Users/BEST LAPTOP/Desktop/STAGE2_color_DETECTION/weights/last.pt')  # from my PC
model = YOLO('weights/last.pt')  #  from Streamlit cloud

# Load the trained Random Forest model
#rf_model_path = "C:/Users/BEST LAPTOP/Desktop/STAGE2_color_DETECTION/weights/RF_model_4colors.pkl"  # from my PC
rf_model_path = "weights/RF_model_4colors.pkl"#  from Streamlit cloud
rf_model = joblib.load(rf_model_path)

# Set a confidence threshold
confidence_threshold = 0.7

# Define a fraction to determine the size of the centered ROI relative to the detected region
roi_fraction = 0.5  # E.g., ROI will be 50% of the detected region's dimensions

# Function to convert BGR to hex
def bgr_to_hex(bgr):
    b, g, r = bgr
    # Ensure the values are integers in the range 0-255
    b, g, r = int(round(b)), int(round(g)), int(round(r))
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# Streamlit app
st.title('Tongue Color Detection with YOLO and RF')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "avif", "webp", "bmp", "tiff"])

if uploaded_file is not None:
    # Read and process the image
    image = np.array(Image.open(uploaded_file))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Perform object detection
    results = model(image_bgr)
    
    # Define colors and thickness
    bbox_color = (255, 0, 0)  # BGR format for blue
    roi_color = (0, 0, 255)   # BGR format for red
    thickness = 2

    # Initialize status and table elements
    status_text = st.empty()
    means_table = st.empty()
    color_box = st.empty()

    roi_found = False  # Flag to check if ROI is detected

    # Loop through the detections and draw bounding boxes and ROIs
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        for box, score in zip(boxes, scores):
            if score >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box)  # Convert to integers
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), bbox_color, thickness)

                # Define ROI dimensions
                width = x2 - x1
                height = y2 - y1
                roi_width = int(width * roi_fraction)
                roi_height = int(height * roi_fraction)
                
                # Calculate ROI coordinates
                roi_x1 = x1 + (width - roi_width) // 2
                roi_y1 = y1 + (height - roi_height) // 2
                roi_x2 = roi_x1 + roi_width
                roi_y2 = roi_y1 + roi_height
                
                # Draw ROI in red
                cv2.rectangle(image_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, thickness)
                
                # Crop the centered ROI
                center_roi_cropped = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_found = True
                
                if center_roi_cropped is not None and center_roi_cropped.size > 0:
                    # Convert the cropped centered ROI to different color spaces
                    yuv = cv2.cvtColor(center_roi_cropped, cv2.COLOR_BGR2YUV)
                    Y, U, V = cv2.split(yuv)

                    hsv = cv2.cvtColor(center_roi_cropped, cv2.COLOR_BGR2HSV)
                    H, S, V = cv2.split(hsv)

                    lab = cv2.cvtColor(center_roi_cropped, cv2.COLOR_BGR2Lab)
                    L, A, B_lab = cv2.split(lab)

                    ycbcr = cv2.cvtColor(center_roi_cropped, cv2.COLOR_BGR2YCrCb)
                    Y1, Cr, Cb = cv2.split(ycbcr)

                    hls = cv2.cvtColor(center_roi_cropped, cv2.COLOR_BGR2HLS)
                    H_hls, L_hls, S_hls = cv2.split(hls)

                    # Convert BGR to YIQ
                    def bgr_to_yiq(bgr):
                        rgb = bgr[..., ::-1]
                        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                        Y = 0.299 * R + 0.587 * G + 0.114 * B
                        I = 0.596 * R - 0.274 * G - 0.321 * B
                        Q = 0.211 * R - 0.523 * G + 0.312 * B
                        return Y, I, Q

                    Y_i, I_i, Q_i = bgr_to_yiq(center_roi_cropped)

                    # Compute the mean intensity for each channel
                    def mean_intensity(channel):
                        return np.mean(channel)

                    # Calculate means for all expected color features
                    means = {
                        'red': mean_intensity(center_roi_cropped[..., 2]),
                        'green': mean_intensity(center_roi_cropped[..., 1]),
                        'blue': mean_intensity(center_roi_cropped[..., 0]),
                        'Y1': mean_intensity(Y1),
                        'cblue': mean_intensity(Cb),
                        'cred': mean_intensity(Cr),
                        'H': mean_intensity(H),
                        'S': mean_intensity(S),
                        'V': mean_intensity(V),
                        'L': mean_intensity(L),
                        'A': mean_intensity(A),
                        'B': mean_intensity(B_lab),
                        'Y': mean_intensity(Y_i),
                        'I': mean_intensity(I_i),
                        'Q': mean_intensity(Q_i)
                    }

                    # Convert the means to a DataFrame with the same column names as used during training
                    df = pd.DataFrame([means])

                    # Predict the class using the trained model
                    prediction = rf_model.predict(df)[0]

                    # Determine color based on prediction
                    color_message = ""
                    if prediction == 1:
                        color_message = "1 - Pink"
                    elif prediction == 2:
                        color_message = "2 - Green"
                    elif prediction == 3:
                        color_message = "3 - Yellow"
                    elif prediction == 4:
                        color_message = "4 - Blue"
                    else:
                        color_message = f"{prediction} - White"

                    # Display the prediction and color box
                    status_text.markdown(f"Predicted class: {color_message}")

                    # Calculate average color for the ROI
                    average_color_bgr = np.mean(center_roi_cropped, axis=(0, 1))
                    average_color_hex = bgr_to_hex(average_color_bgr)

                    # Display the color box with the average color
                    color_box.markdown(
                        f'<div style="width: 100px; height: 100px; background-color: {average_color_hex}; border: 2px solid black; margin-top: 10px;"></div>',
                        unsafe_allow_html=True
                    )

                    # Display the means as a table
                    means_df = pd.DataFrame([means])
                    means_table.dataframe(means_df)

                    # Save the DataFrame to an Excel file in memory
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    excel_buffer.seek(0)

                    # Provide the download button
                    st.download_button(
                        label="Download Mean Intensity Values",
                        data=excel_buffer,
                        file_name='mean_intensity_values.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    if not roi_found:
        status_text.text("No ROI detected")

    # Convert the image from BGR to RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes and ROIs using Streamlit
    st.image(image_rgb, channels="RGB")

# Add footer with custom text
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f1f1f1;
        border-top: 1px solid #ddd;
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
        Created by Dr. Ali Al-Naji
    </div>
    """,
    unsafe_allow_html=True
)

