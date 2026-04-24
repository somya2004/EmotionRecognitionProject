import cv2
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model("saved_model/emotion_model.h5")

# Emotion Labels
labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load Face Detector
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = face / 255.0
        face = face.reshape(1,48,48,1)

        prediction = model.predict(face, verbose=0)
        emotion = labels[np.argmax(prediction)]

        # Draw Rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Show Emotion
        cv2.putText(
            frame,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()