from imutils import paths
import face_recognition
import pickle
import cv2
import os
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
import multiprocessing

# Configure logging
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Main script
print("-----------------------------------------")
print("-------- Searching Datasets -------------")
print("-----------------------------------------")
imagePaths = list(paths.list_images("Dataset"))
print(f"Found {len(imagePaths)} images in the dataset.")

if not imagePaths:
    print("-----------------------------------------")
    print("----------- No Datasets Found -----------")
    print("-----------------------------------------")
    print("Exiting.............")
    exit()

print("We found data successfully. Starting processing...")

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

print("Processing complete.")
