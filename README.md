# Face-Emotion-detection
A deep learning-based project to detect human emotions from facial expressions using Convolutional Neural Networks (CNNs).

🚀 Overview
This project uses a CNN model trained on grayscale face images to classify emotions such as:

😃 Happy

😢 Sad

😡 Angry

😲 Surprise

😐 Neutral

😱 Fear

🤢 Disgust

# Technologies Used
-Python

-TensorFlow / Keras

-OpenCV – for real-time face detection

-NumPy, Pandas – data handling

-Matplotlib – visualization


# Dataset

-Grayscale images of shape 48x48

-Corresponding emotion labels

# Project Structure
📂 Face-Emotion-Detection/
├── 📄 model.py           # CNN architecture
├── 📄 train.py           # Model training script
├── 📄 test.py            # Evaluation and accuracy
├── 📄 real_time.py       # Live webcam emotion detection
├── 📁 data/              # Dataset directory
├── 📁 saved_model/       # Trained model
├── 📄 requirements.txt   # Required Python libraries

# Model Architecture
Conv2D → ReLU → MaxPooling  
Conv2D → ReLU → MaxPooling  
Flatten → Dense → Dropout  
Dense (Softmax)
Input shape: (48, 48, 1)

Output: 7 emotion classes



