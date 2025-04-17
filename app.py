import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace

st.set_page_config(layout="wide")
st.title("⚡ Fast Real-Time Face Analysis")
st.markdown("**Detects Age, Gender, and Mood from Webcam with Performance Optimization**")

debug_mode = st.checkbox("🪛 Enable Debug Mode")
FRAME_WINDOW = st.image([])

# Initialize FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Use CAP_ANY for best webcam backend
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_emotion = "Detecting..."
last_emotion_scores = {}
last_age = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("No webcam input!")
        break

    small_frame = cv2.resize(frame, (320, 240))
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = app.get(frame_rgb)

    if faces:
        main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        box = main_face.bbox.astype(int)

        # Scale box to original frame
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        box = (box * np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)

        gender = "Male" if main_face.sex == 'M' else "Female"

        # Crop from full frame
        face_crop_high_res = frame[box[1]:box[3], box[0]:box[2]]

        if frame_count % 10 == 0:  # Analyze every 10 frames
            try:
                result = DeepFace.analyze(face_crop_high_res, actions=['emotion', 'age'], enforce_detection=False)
                last_emotion = result[0]['dominant_emotion']
                last_emotion_scores = result[0]['emotion']
                last_age = int(result[0]['age'])
            except:
                last_emotion = "Unknown"
                last_emotion_scores = {}
                last_age = "N/A"

        label = f"{gender}, Age: ~{last_age}, Mood: {last_emotion}"

        if debug_mode:
            label += f"\n[Gender Raw: {main_face.sex}]"
            label += f"\n[Emotions: {last_emotion_scores}]"

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        for i, line in enumerate(label.split("\n")):
            cv2.putText(frame, line, (box[0], box[1] - 10 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_count += 1

