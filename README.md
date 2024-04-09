# DeepEYE: Anomaly Detection in Video Surveillance

Welcome to the DeepEYE: Anomaly Detection in Video Surveillance project repository! This repository contains tools and resources for detecting anomalies in video surveillance footage. Whether you're a data scientist, a machine learning enthusiast, or a developer, this project aims to provide you with a comprehensive solution for anomaly detection in video data.

## Project Structure

### Data
- **Train_data**: This directory contains training data for building anomaly detection models.
- **Test_Data**: This directory contains test data for evaluating the performance of the anomaly detection models.

### Code
- **vid2array.py**: This Python script implements the data preparation.
- **model.py**: This script implements the anomaly detection model.
- **test.py**: This script tests the anomaly detection model.
- **training.npy**: Numpy file containing preprocessed training data.
- **model.keras**: Trained anomaly detection model saved in keras format.
- **app.py**: A real-time application showcasing how to use the anomaly detection model.
- **requirements.txt**: List of Python dependencies required to run the code.

## How to Use
1. Clone this repository to your local machine:

```bash
git clone https://github.com/jaigane6387/DeepEYE_Video_Surviellience.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3.  Prepare the data by converting all the videos to frames:
```bash
python vid2array.py
```
4. Train the anomaly detection model using the provided training data
```bash
python model.py
```
5. Test the trained model on test data:
```bash
python test.py
```

6. To view LiveExample, you can run the Live example application:
```bash
python app.py
```
## Output

<!-- Example GIF -->
![Anomaly Detection Example](output/anomaly_detection.gif)

<!-- Example Image -->
![Anomaly Detection Result](output/anomaly_detection_result.png)

## Tech Stack

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" width="100" height="100"/>
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="TensorFlow" width="100" height="100"/>
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" alt="OpenCV" width="100" height="100"/>
</div>

These technologies are the backbone of this project, providing powerful tools for Data Manipulation, Deep Learning, and Computer Vision tasks.

## Bug Reports and Feature Requests

If you encounter any issues with the code or have suggestions for new features, please don't hesitate to [open an issue](https://github.com/yourusername/Video-Surveillance-Anomaly-Event-Detection/issues). Your feedback is highly appreciated and will help improve this project for everyone.

### Social
Connect with me on social media platforms to stay updated on this project and more!

- Blogs: https://dataaspirant.com/author/jaiganesh-nagidi/
- LinkedIn: https://www.linkedin.com/in/jai-ganesh-nagidi/
- Kaggle: https://www.kaggle.com/jaiganeshnagidi

