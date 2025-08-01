import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained emotion detection model
model = load_model('emotiomdetection.h5')


# Emotion labels corresponding to model output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess face for model
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float') / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(roi_reshaped)
        label = emotion_labels[np.argmax(prediction)]

        # Display label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()




