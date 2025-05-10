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
import logging
import os
from joblib import Parallel, delayed
from imutils import paths
import multiprocessing
from tqdm import tqdm


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



# ======================== Utility Functions ========================




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

with open("./FaceRec_Trained_Model.pickle", "rb") as f:
    puckledata = pickle.load(f)
    knownEncodings = puckledata["encodings"]
    knownNames = puckledata["names"]
    logging.error(knownNames)


    
def analyze_pickle_structure(pickle_path):
    """Analyze and print the structure of the pickle file to help debugging"""
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        print("\n=== PICKLE FILE ANALYSIS ===")
        if isinstance(data, dict):
            print(f"Type: Dictionary with {len(data.keys())} keys")
            print(f"Keys: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  '{key}': List with {len(value)} items")
                    if value and len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            print(f"    First item shape: {value[0].shape}")
                else:
                    print(f"  '{key}': {type(value)}")
        elif isinstance(data, list):
            print(f"Type: List with {len(data)} items")
            if data and len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], np.ndarray):
                    print(f"First item shape: {data[0].shape}")
        else:
            print(f"Type: {type(data)}")
        print("===========================\n")
        return data
    except Exception as e:
        print(f"Error analyzing pickle file: {e}")
        return None



def recognize_faces(image):
    logging.info("Connected with Data")
    
    """Detects and recognizes faces in an image."""
    img_resized = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(img_resized, model="hog")
    encodes_cur_frame = face_recognition.face_encodings(img_resized, faces_cur_frame)

    name = "Unknown"

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        face_distances = face_recognition.face_distance(knownEncodings, encode_face)
        match_index = np.argmin(face_distances)

        if face_distances[match_index] < 0.6:
            name = knownNames[match_index].title()

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

# Function to save images from API
def save_images_from_api(images: List[UploadFile], username: str):
    save_dir = os.path.join("Dataset", username)
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        file_path = os.path.join(save_dir, f"{username}_{i}.jpg")
        with open(file_path, "wb") as buffer:
            buffer.write(image.file.read())

# Function to process a single image
def process_image(imagePath):
    try:
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        
        # Skip if the image is corrupted or invalid
        if image is None:
            logging.warning(f"Skipping corrupted image: {imagePath}")
            return [], []

        # Detect faces in the image
        boxes = face_recognition.face_locations(image, model='cnn')  # Use 'cnn' for better accuracy (requires GPU)
        encodings = face_recognition.face_encodings(image, boxes)

        # If no faces are detected, skip the image
        if not encodings:
            logging.warning(f"No faces detected in image: {imagePath}")
            return [], []

        # Return encodings and corresponding names
        return encodings, [name] * len(encodings)
    except Exception as e:
        logging.error(f"Error processing {imagePath}: {e}")
        return [], []

# Endpoint to register faces
@app.post("/registerfaces/", summary="Register Faces", tags=["Face Recognition"])
async def registerfaces(username: str = Form(...), images: List[UploadFile] = File(...)):
    """
    Endpoint to register faces by receiving a list of images.
    Expecting 10 images for a single user.
    """
    if len(images) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 images required")

    # Save uploaded images
    save_images_from_api(images, username)

    # Process images to generate face encodings
    imagePaths = list(paths.list_images(os.path.join("Dataset", username)))
    if not imagePaths:
        raise HTTPException(status_code=400, detail="No images found for processing")

    # Use all available CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(process_image)(imagePath) for imagePath in tqdm(imagePaths, desc="Processing Images"))

    # Combine results
    knownEncodings = []
    knownNames = []
    for encodings, names in results:
        knownEncodings.extend(encodings)
        knownNames.extend(names)

    # Save encodings to disk
    if knownEncodings:
        print("[INFO] Serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        with open('FaceRec_Trained_Model.pickle', "wb") as f:
            pickle.dump(data, f)
        print("[INFO] Encodings saved to FaceRec_Trained_Model.pickle")
    else:
        print("[INFO] No encodings were generated. Check the dataset and logs for errors.")

    return {"message": f"Received 10 images for {username} and processed successfully"}

# Helper function to list image paths
def list_images(directory: str):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
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

             
                # Perform face recognition
                processed_image, name = recognize_faces(image)

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
