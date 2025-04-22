import cv2
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# custom transformer for centering faces based on mean face
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        if not hasattr(self, "mean_face"):
            raise AttributeError("MeanCentering has no attribute 'mean_face'. Did you fit the transformer?")
        return X - self.mean_face

# load the trained model pipeline
with open("eigenface_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

# load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_size = (128, 128)

# detect faces in a grayscale image
def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    return face_cascade.detectMultiScale(image_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

# crop detected faces and store their coordinates
def crop_faces(image_gray, faces):
    cropped_faces = []
    coords = []

    for x, y, w, h in faces:
        cropped = image_gray[y:y+h, x:x+w]
        cropped_faces.append(cropped)
        coords.append((x, y, w, h))
    
    return cropped_faces, coords

# resize face image and flatten to 1D array
def resize_and_flatten(face):
    resized = cv2.resize(face, face_size)
    return resized.flatten()

# compute eigenface classification score
def get_eigenface_score(X):
    X_pca = pipe[:2].transform(X)
    return np.max(pipe[2].decision_function(X_pca), axis=1)

# draw label and score above the detected face
def draw_text(image, label, score, pos=(0, 0), color=(0, 255, 0)):
    x, y = pos
    text = f"{label} ({score:.2f})"
    cv2.rectangle(image, (x, y - 20), (x + 200, y), color, -1)
    cv2.putText(image, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# recognize faces and annotate them on the image
def recognize_faces(image_gray, image_color):
    faces = detect_faces(image_gray)
    cropped_faces, coords = crop_faces(image_gray, faces)

    if not cropped_faces:
        return image_color

    X_faces = [resize_and_flatten(face) for face in cropped_faces]
    X_faces = np.array(X_faces)
    labels = pipe.predict(X_faces)
    scores = get_eigenface_score(X_faces)

    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(image_color, label, score, (x, y))
    
    return image_color

# start webcam stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = recognize_faces(gray, frame.copy())

    cv2.imshow("Real-Time Face Recognition", output)

    # press 'q' to quit the webcam
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
