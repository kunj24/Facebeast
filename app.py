import streamlit as st
import cv2
import numpy as np
import time
import collections
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model and labels
MODEL_PATH = "model.h5"
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
model = load_model(MODEL_PATH)

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("🎭 Real-Time Facial Emotion Detection")
st.write("Upload an image or use your webcam for real-time emotion detection.")

# Upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# Webcam activation
use_webcam = st.checkbox("Use Webcam")

# Store last 20 predictions to determine the most frequent emotion
predicted_emotions = collections.deque(maxlen=20)

def predict_emotion(image):
    """Predicts emotion from a given image and tracks it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)[0]
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Store the prediction in the deque
        predicted_emotions.append(label)

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, f"{label} ({confidence*100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    processed_img = predict_emotion(img)
    st.image(processed_img, channels="BGR")

if use_webcam:
    st.write("Webcam is running. Press **Stop** to exit.")
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()  # Placeholder for updating frames
    stop_button = st.button("Stop Camera")  # Stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break  # Exit loop when stop button is pressed

        processed_frame = predict_emotion(frame)
        frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    cap.release()
    st.write("Camera stopped.")

    # **Show the most detected mood after stopping**
    if predicted_emotions:
        most_common_mood = collections.Counter(predicted_emotions).most_common(1)[0][0]
        st.success(f"**Most Detected Mood:** {most_common_mood}")
    else:
        st.warning("No emotions detected.")
