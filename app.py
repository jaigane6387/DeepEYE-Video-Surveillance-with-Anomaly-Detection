import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import imutils
import json
import tempfile
import os

def mean_squared_loss(x1, x2):
    difference = x1 - x2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_difference = difference ** 2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance / n_samples

    return mean_distance

# Load the Keras model
model = load_model("model\\aved_model.keras")

# Function to perform anomaly detection on video frames and save abnormal frames
def detect_anomalies(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    imagedump = []
    abnormal_frames = []

    # Create a directory to save abnormal frames
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        image = imutils.resize(frame, width=700, height=600)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)

        if frame_count % 10 == 0:
            imagedump = np.array(imagedump)
            imagedump.resize(227, 227, 10)
            imagedump = np.expand_dims(imagedump, axis=0)
            imagedump = np.expand_dims(imagedump, axis=4)

            output = model.predict(imagedump)

            loss = mean_squared_loss(imagedump, output)

            if loss > 0.00038:
                st.error('🚨 Abnormal Event Detected 🚨')
                st.image(image, caption="", channels="BGR")
                abnormal_frames.append(frame_count)
                # Save abnormal frame as an image
                cv2.imwrite(os.path.join(results_dir, f"abnormal_frame_{frame_count}.jpg"), image)

            imagedump = []

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save abnormal frames indices to JSON
    with open(os.path.join(results_dir, 'abnormal_frames.json'), 'w') as json_file:
        json.dump(abnormal_frames, json_file)

# Streamlit code for file upload and anomaly detection
st.markdown("<h1 style='text-align: center; color: #006699;'>DeepEYE Anomaly Surveillance 👁️</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    detect_anomalies(tmp_file_path)

    os.unlink(tmp_file_path)
