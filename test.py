import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import imutils

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance

model = load_model("model\\saved_model.keras")

# Replace the test video data path here
cap = cv2.VideoCapture("C:\\Users\\jaiga\\OneDrive\\Desktop\\new\\My_Projects\\DeepEYE\\Avenue_Dataset\\testing_videos\\test1.mp4")
print(cap.isOpened())

while cap.isOpened():
    im_frames = []
    ret, frame = cap.read()

    if not ret:
        break

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        image = imutils.resize(frame, width=700, height=600)

        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

    im_frames = np.array(im_frames)
    im_frames.resize(227, 227, 10)
    im_frames = np.expand_dims(im_frames, axis=0)
    im_frames = np.expand_dims(im_frames, axis=4)

    output = model.predict(im_frames)
    loss = mean_squared_loss(im_frames, output)
    print("Mean Squared Loss:", loss)

    if frame is None:
        print("Frame is None")

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if 0.00062 < loss < 0.00067:  # Adjusted condition to use range
        print('Abnormal Event Detected')
        # Draw bounding box around the abnormal event
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
        
        text = "Abnormal Event"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        # Draw a white rectangle behind the text
        cv2.rectangle(image, (50, 50 - text_height), (50 + text_width, 50), (255, 255, 255), -1)
        
        cv2.putText(image, text, (45, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    resized_frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("DeepEYE Anomaly Surveillance", resized_frame)

cap.release()
cv2.destroyAllWindows()