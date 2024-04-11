import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

image_data = []
# replace the path with your training data path
train_path = "C:\\Users\\jaiga\\DeepEYE\\train_data"
fps = 5
train_videos = os.listdir(train_path)
train_images_path = train_path + '/frames'

# Create frames directory if it doesn't exist
if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

def data_store(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image_data.append(gray)

for video in train_videos:
    video_path = os.path.join(train_path, video)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0

    while success:
        frame_path = os.path.join(train_images_path, f"{count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        count += fps  # Adjust this if you want to capture frames at a different rate
    cap.release()

    images = os.listdir(train_images_path)

    for image in images:
        image_path = os.path.join(train_images_path, image)
        data_store(image_path)

image_data = np.array(image_data)
a, b, c = image_data.shape
image_data.resize(b, c, a)
image_data = (image_data - image_data.mean()) / (image_data.std())
image_data = np.clip(image_data, 0, 1)

# Storing the data 
np.save('training.npy', image_data)
