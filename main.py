from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import base64
import pickle
import json
import face_recognition
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import distance as dist
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="An API for face recognition, liveness detection, and image processing.",
    version="1.0.0",
)

# Ensure the Dataset folder exists
DATASET_DIR = Path("Dataset")
DATASET_DIR.mkdir(exist_ok=True)

# CORS Middleware for allowing all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained face recognition model
with open("FaceRec_Trained_Model.pickle", "rb") as f:
    data = pickle.load(f)

knownEncodeList = data["encodings"]
classNames = data["names"]

# ======================== Utility Functions ========================
def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) for liveness detection."""
    A, B, C = dist.euclidean(eye[1], eye[5]), dist.euclidean(eye[2], eye[4]), dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0  # Avoid division by zero

def liveness_detection(image):
    """Detects liveness based on eye aspect ratio (EAR)."""
    face_landmarks = face_recognition.face_landmarks(image)

    if not face_landmarks:
        return False, image

    for landmarks in face_landmarks:
        leftEAR = eye_aspect_ratio(landmarks["left_eye"])
        rightEAR = eye_aspect_ratio(landmarks["right_eye"])
        ear = (leftEAR + rightEAR) / 2.0

        # Overlay EAR value on image for debugging
        cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if ear < 0.25:  # Threshold for closed eyes
            return True, image

    return False, image

def save_images_from_api(images: List[UploadFile], username: str):
    user_dir = DATASET_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        img_array = np.frombuffer(image.file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail=f"Invalid image: {image.filename}")

        # Convert to grayscale & resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray, (400, 400))

        # Save image
        img_path = user_dir / f"file_{idx}.jpg"
        cv2.imwrite(str(img_path), resized_img)
        print(f"Saved {img_path}")

    return {"message": f"Saved {len(images)} images for {username}"}

def recognize_faces(image):
    """Detects and recognizes faces in an image."""
    img_resized = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(img_resized, model="hog")
    encodes_cur_frame = face_recognition.face_encodings(img_resized, faces_cur_frame)

    name = "Unknown"

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        face_distances = face_recognition.face_distance(knownEncodeList, encode_face)
        match_index = np.argmin(face_distances)

        if face_distances[match_index] < 0.6:
            name = classNames[match_index].title()

        # Scale coordinates back to original image size
        y1, x2, y2, x1 = [val * 4 for val in face_loc]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (128, 0, 128), 1)

    return image, name

# ============================== Pydantic Models =====================

class RegisterFacesRequest(BaseModel):
    images: List[UploadFile]  # Expecting a list of images
    username: str

class RegisterFacesResponse(BaseModel):
    message: str

class WebSocketResponse(BaseModel):
    image: str
    name: str

# ============================== Normal API ===========================

@app.post("/registerfaces/", response_model=RegisterFacesResponse, summary="Register Faces", tags=["Face Recognition"])
async def registerfaces(username: str = Form(...), images: List[UploadFile] = File(...)):
    """
    Endpoint to register faces by receiving a list of images.
    Expecting 10 images for a single user.
    """
    if len(images) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 images required")
    save_images_from_api(images, username)

    return {"message": f"Received 10 images for {username}"}
# ======================== WebSocket Endpoint ========================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles real-time face recognition and liveness detection over WebSocket."""
    await websocket.accept()
    await websocket.send_json({"status": "Connection established"})

    try:
        while True:
            try:
                data = await websocket.receive_text()
                image_data = base64.b64decode(data)

                # Decode image
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    await websocket.send_json({"status": "Error: Unable to decode image"})
                    continue

                # Perform liveness detection
                is_live, image_with_liveness = liveness_detection(image)
                if not is_live:
                    await websocket.send_json({"status": "Liveness detection failed"})
                    continue

                # Perform face recognition
                processed_image, name = recognize_faces(image_with_liveness)

                # Convert processed image to base64
                _, buffer = cv2.imencode(".jpg", processed_image)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_json({"image": image_base64, "name": name})

            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                await websocket.send_json({"status": error_msg})
                print(error_msg)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
