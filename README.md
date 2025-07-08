# deepfake-video-detection
This project is a Deep Learning-based Deepfake Video Detection System, developed to identify manipulated or synthetically generated videos using face-based analysis and machine learning techniques. It leverages CNN and temporal models to detect inconsistencies in video frames and facial features.

🔍 Project Overview
With the rise of AI-generated media, deepfakes have become a serious threat to privacy, identity, and digital trust. This system helps in detecting such forgeries by analyzing facial landmarks and video frames using a trained model.

🧠 Core Features
🔎 Face detection and alignment using MTCNN

🎥 Frame extraction from videos

🧬 Deep Learning model for classification (Fake vs Real)

📊 Evaluation metrics like accuracy, precision, recall

🗃️ Dataset compatibility with FaceForensics++, DFDC, etc.

🛠️ Tech Stack
Programming Language: Python

Libraries:

OpenCV

NumPy

TensorFlow / PyTorch (based on model used)

MTCNN for face detection

Keras (if used in your model)

Jupyter Notebooks for training and evaluation

📂 Folder Structure
bash
Copy
Edit
Deepfake_video_detection-main/
├── dataset/             # For storing videos
├── models/              # Pretrained or trained models
├── utils/               # Helper scripts (frame extraction, etc.)
├── notebooks/           # Jupyter notebooks for training/testing
├── requirements.txt     # Python dependencies
├── train.py             # Model training script
├── test.py              # Model evaluation script
└── README.md
⚙️ How to Use
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/your-username/Deepfake_video_detection.git
cd Deepfake_video_detection
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare Dataset:

Place your videos (real and fake) in the dataset/ folder.

Run the frame extraction script to preprocess.

Train the Model:

bash
Copy
Edit
python train.py
Test or Predict:

bash
Copy
Edit
python test.py
📈 Evaluation
Model performance is measured using:

Accuracy

Precision

Recall

Confusion Matrix

ROC-AUC Curve

📸 Sample Results
(Add plots or confusion matrices here if available)

🤝 Contributors
Pranam Gowda

[Add collaborators if any]

📜 License
This project is licensed under the MIT License.

