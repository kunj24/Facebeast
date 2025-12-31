import cv2
import numpy as np
import time
import webbrowser
from collections import Counter
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import gradio as gr

# Define constants
IMG_SIZE = (48, 48)
MODEL_PATH = "model.h5"

# Emotion categories
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Updated Emotion-based YouTube Music Playlist Mapping
emotion_playlists = {
    "Happy": "https://music.youtube.com/watch?v=OPf0YbXqDm0&list=RDCLAK5uy_lV2nSIwNU8070lncN33m1P_VbWwkGKxzE",
    "Sad": "https://music.youtube.com/watch?v=yGi1MePEN-k",
    "Neutral": "https://music.youtube.com/playlist?list=RDCLAK5uy_kskrFUGb5Tnz3-x4wyK9Q5j8RgfwQvq4k",
    "Angry": "https://music.youtube.com/playlist?list=OLAK5uy_ljpWIfGZ8cRxuh9NKRD4wx64o2cY7-dDg",
    "Surprise": "https://music.youtube.com/playlist?list=RDCLAK5uy_n8uk1Fcot716E7mchDmOSViUixZn-FwlQ"
}

# Function to open a YouTube Music playlist
def redirect_to_playlist(emotion): 
    playlist_url = emotion_playlists.get(emotion)
    if playlist_url:
        print(f"🎵 Opening YouTube Music Playlist for {emotion} mood: {playlist_url}")
        return playlist_url
    else:
        print("No playlist found for this emotion.")
        return None

# Load trained model
model = load_model(MODEL_PATH)

# Load Haar cascade classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect emotion in an image
def detect_emotion(image):
    # Enhance image for better face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    
    # Try multiple detection passes with different parameters
    faces = face_classifier.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # More sensitive
        minNeighbors=2,    # Very lenient
        minSize=(10, 10),  # Detect very small faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    predictions = []
    output_image = image.copy()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        predictions.append(label)

        # Draw rectangle and label
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(output_image, f"{label} ({confidence*100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if predictions:
        most_common_emotion = Counter(predictions).most_common(1)[0][0]
        return output_image, most_common_emotion
    else:
        return output_image, "No face detected"

# Custom CSS for modern UI
custom_css = """
body {
    background: linear-gradient(135deg, #1e1e2f, #2b4162, #121212);
    color: white;
    font-family: 'Poppins', sans-serif;
    text-align: center;
}

.gradio-container {
    max-width: 600px;
    margin: auto;
    padding: 30px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(255, 255, 255, 0.1);
}

h1 {
    color: #ffcc00;
    text-align: center;
    font-size: 30px;
}

.gr-button {
    background-color: #ff6600 !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    padding: 10px 20px !important;
}

.gr-textbox, .gr-file, .gr-video {
    border-radius: 10px !important;
    border: 2px solid #ffcc00 !important;
    padding: 10px !important;
    font-size: 16px !important;
}
"""

# Gradio Blocks interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>Real-Time Facial Emotion Detection</h1>")
    gr.Markdown("Upload an image or use your webcam to detect emotions and get a YouTube Music playlist recommendation!")
    
    with gr.Tab("Webcam Analysis"):
        gr.Markdown("📹 Capture a photo from your webcam to detect emotion")
        webcam_input = gr.Image(label="Webcam", sources=["webcam"], type="pil")
        webcam_output = gr.Image(label="Processed Image")
        webcam_emotion = gr.Textbox(label="Detected Emotion")
        webcam_playlist = gr.Markdown(label="🎵 Music Playlist")
        
        def webcam_analysis(image):
            if image is None:
                return None, "No image captured", "Upload an image to get started"
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output_image, emotion = detect_emotion(image)
            if emotion != "No face detected":
                playlist_url = redirect_to_playlist(emotion)
                if playlist_url:
                    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), emotion, f"🎵 **[Open {emotion} Playlist on YouTube Music]({playlist_url})**"
            return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), emotion, "No emotion detected"
        
        webcam_input.change(
            webcam_analysis,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_emotion, webcam_playlist]
        )
    
    with gr.Tab("Image Analysis"):
        image_input = gr.Image(label="Upload Image", type="pil")
        image_output = gr.Image(label="Processed Image")
        image_emotion = gr.Textbox(label="Detected Emotion")
        image_playlist = gr.Markdown(label="🎵 Music Playlist")
        
        def image_analysis(image):
            if image is None:
                return None, "No image uploaded", "Upload an image to get started"
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output_image, emotion = detect_emotion(image)
            if emotion != "No face detected":
                playlist_url = redirect_to_playlist(emotion)
                if playlist_url:
                    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), emotion, f"🎵 **[Open {emotion} Playlist on YouTube Music]({playlist_url})**"
            return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), emotion, "No emotion detected"
        
        image_input.change(
            image_analysis,
            inputs=image_input,
            outputs=[image_output, image_emotion, image_playlist]
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
