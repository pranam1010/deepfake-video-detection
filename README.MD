 Overview
This project leverages deep learning techniques to detect deepfake videos by analyzing subtle inconsistencies in facial features and motion. It uses pre-trained models and facial landmarks to classify video frames as real or fake with high accuracy.

🎯 Objective
To create an automated system that can analyze video frames and identify synthetic manipulations typically found in deepfake media.

🔍 Key Features
Face detection and alignment

Frame-by-frame deepfake prediction

Visualization of classification results

Utilization of OpenCV and Deep Learning models

Live camera support for real-time analysis

🧠 Tech Stack
Python

Libraries: OpenCV, Keras, TensorFlow, NumPy

Pretrained CNN Models

Dlib for facial landmark detection

📁 Structure
dataset/ – Sample deepfake and real videos

model/ – Pretrained model files

predict.py – Frame analysis and prediction

utils.py – Helper functions for preprocessing and face detection
