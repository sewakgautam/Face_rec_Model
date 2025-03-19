import cv2
import numpy as np
import pickle
import dlib
import os
import time

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

def setup_face_recognition_system(pickle_path=None, threshold=0.6):
    """
    Set up and run the face recognition system
    
    Args:
        pickle_path: Path to the pickle file containing face encodings
        threshold: Recognition threshold (lower is stricter)
        use_test_data: If True, create and use test data instead
    """

    # Use default path if not specified
    if pickle_path is None:
        pickle_path = "FaceRec_Trained_Model.pickle"
    
    # Analyze pickle structure first
    print(f"Loading face data from: {pickle_path}")
    data = analyze_pickle_structure(pickle_path)
    
    # Initialize with empty data
    known_face_encodings = []
    known_face_names = []
    
    # Try to extract face data directly
    if data is not None:
        try:
            # Attempt different potential structures based on the analysis
            if isinstance(data, dict):
                potential_encoding_keys = ["encodings", "encoding", "faces", "embeddings", "descriptors"]
                potential_name_keys = ["names", "name", "labels", "ids"]
                
                # Find encoding key
                encoding_key = None
                for key in potential_encoding_keys:
                    if key in data:
                        encoding_key = key
                        break
                
                # Find name key
                name_key = None
                for key in potential_name_keys:
                    if key in data:
                        name_key = key
                        break
                
                if encoding_key and name_key:
                    known_face_encodings = data[encoding_key]
                    known_face_names = data[name_key]
                    print(f"Found encodings under key '{encoding_key}' and names under key '{name_key}'")
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                # Try to parse as [encodings, names]
                known_face_encodings = data[0]
                known_face_names = data[1]
            
            # Print what we found
            print(f"Extracted {len(known_face_encodings)} face encodings and {len(known_face_names)} names")
            if known_face_names:
                print(f"Example names: {known_face_names[:min(3, len(known_face_names))]}")
            
            # Verify the data structure
            if len(known_face_encodings) == 0 or len(known_face_names) == 0:
                print("WARNING: No face encodings or names found in the pickle file!")
            elif len(known_face_encodings) != len(known_face_names):
                print(f"WARNING: Mismatch between number of encodings ({len(known_face_encodings)}) and names ({len(known_face_names)})!")
        except Exception as e:
            print(f"Error extracting data from pickle: {e}")
    
    # Initialize the face detector
    print("Initializing face detection models...")
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = None
    face_rec_model = None
    
    # Try to load the dlib models if available
    try:
        # Check for shape predictor
        shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(shape_predictor_path):
            face_predictor = dlib.shape_predictor(shape_predictor_path)
            print(f"Loaded shape predictor from {shape_predictor_path}")
        else:
            print(f"WARNING: Shape predictor file not found: {shape_predictor_path}")
            print("Face recognition will not work without this file!")
        
        # Check for face recognition model
        face_rec_path = "dlib_face_recognition_resnet_model_v1.dat"
        if os.path.exists(face_rec_path):
            face_rec_model = dlib.face_recognition_model_v1(face_rec_path)
            print(f"Loaded face recognition model from {face_rec_path}")
        else:
            print(f"WARNING: Face recognition model file not found: {face_rec_path}")
            print("Face recognition will not work without this file!")
    except Exception as e:
        print(f"Error loading dlib models: {e}")
    
    # Initialize webcam
    print("Initializing camera...")
    video_capture = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create a detection area
    _, frame = video_capture.read()
    if frame is None:
        print("Failed to grab initial frame. Check camera connection.")
        video_capture.release()
        return
        
    frame_height, frame_width = frame.shape[:2]
    
    # Define detection area (centered rectangle covering ~40% of frame)
    margin_x = int(frame_width * 0.2)
    margin_y = int(frame_height * 0.2)
    detection_area = {
        'x1': margin_x,
        'y1': margin_y,
        'x2': frame_width - margin_x,
        'y2': frame_height - margin_y,
    }
    
    print("\n=== Face Recognition System Starting ===")
    print(f"Recognition threshold: {threshold} (lower is stricter)")
    print("Place your face in the highlighted green rectangle")
    print("Press 'q' to quit, 't' to adjust threshold, 'd' for debug info")
    
    # Debug mode flag
    debug_mode = False
    last_detection_time = time.time()
    detection_timeout = 2  # seconds between detection messages
    
    # Main processing loop
    while True:
        # Capture frame from webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Make a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Draw the detection area rectangle
        cv2.rectangle(display_frame, 
                     (detection_area['x1'], detection_area['y1']), 
                     (detection_area['x2'], detection_area['y2']), 
                     (0, 255, 0), 2)
        
        # Convert the image for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use dlib's face detector
        dlib_faces = face_detector(gray, 1)  # 1 = upsample once for better detection
        
        person_in_area = "Unknown"
        detection_reason = ""
        
        # Process each detected face
        for i, face in enumerate(dlib_faces):
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            
            # Calculate face center
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            
            # Check if face is within detection area
            in_detection_area = (detection_area['x1'] <= face_center_x <= detection_area['x2'] and
                                detection_area['y1'] <= face_center_y <= detection_area['y2'])
            
            # Get face encoding
            face_encoding = None
            if face_predictor and face_rec_model:
                try:
                    shape = face_predictor(rgb_frame, face)
                    face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
                    
                    if debug_mode:
                        print(f"Face #{i+1} encoding shape: {face_encoding.shape}")
                except Exception as e:
                    print(f"Error computing face descriptor: {e}")
                    detection_reason = "Error computing face descriptor"
            else:
                detection_reason = "Missing face models"
            
            # Try to recognize the face
            name = "Unknown"
            match_distance = None
            
            if face_encoding is not None and len(known_face_encodings) > 0:
                try:
                    # Compute distances to all known faces
                    distances = []
                    for j, enc in enumerate(known_face_encodings):
                        # Make sure encodings are compatible
                        if len(face_encoding) == len(enc):
                            distance = np.linalg.norm(face_encoding - enc)
                            distances.append(distance)
                            if debug_mode and time.time() - last_detection_time > detection_timeout:
                                if j < 3 or j == len(known_face_encodings) - 1:  # Show first 3 and last match
                                    print(f"Distance to {known_face_names[j]}: {distance:.4f}")
                        else:
                            print(f"Encoding size mismatch: {len(face_encoding)} vs {len(enc)}")
                            distances.append(float('inf'))
                            detection_reason = "Encoding size mismatch"
                    
                    if distances:
                        best_match_index = np.argmin(distances)
                        best_match_distance = distances[best_match_index]
                        
                        if debug_mode and time.time() - last_detection_time > detection_timeout:
                            print(f"Best match: {known_face_names[best_match_index]} with distance {best_match_distance:.4f}")
                            last_detection_time = time.time()
                        
                        if best_match_distance < threshold:
                            name = known_face_names[best_match_index]
                            match_distance = best_match_distance
                        else:
                            detection_reason = f"Distance {best_match_distance:.2f} > threshold {threshold}"
                except Exception as e:
                    print(f"Error during face matching: {e}")
                    detection_reason = "Error during matching"
            elif face_encoding is None:
                detection_reason = "Could not compute face encoding"
            elif len(known_face_encodings) == 0:
                detection_reason = "No known faces in database"
            
            # Set colors based on whether face is in area and recognized
            if in_detection_area:
                color = (0, 255, 0)  # Green
                if name != "Unknown":
                    person_in_area = name
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box around face
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with name and distance (if available)
            label = name
            if match_distance is not None:
                label += f" ({match_distance:.3f})"
                
            cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(display_frame, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display status at the bottom of the frame
        if len(dlib_faces) > 0 and person_in_area != "Unknown":
            status = f"DETECTED: {person_in_area}"
            status_color = (0, 255, 0)  # Green
        elif len(dlib_faces) > 0:
            status = "UNKNOWN PERSON"
            if debug_mode:
                status += f" ({detection_reason})"
            status_color = (0, 165, 255)  # Orange
        else:
            status = "No face detected"
            status_color = (0, 0, 255)  # Red
            
        # Display status message
        cv2.putText(display_frame, status, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Display number of loaded faces and current threshold
        cv2.putText(display_frame, f"Known faces: {len(known_face_encodings)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Threshold: {threshold:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if debug_mode:
            cv2.putText(display_frame, "DEBUG MODE ON", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition System', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
        elif key == ord('t'):
            new_threshold = input("Enter new threshold (current: {:.2f}): ".format(threshold))
            try:
                threshold = float(new_threshold)
                print(f"Threshold set to {threshold}")
            except ValueError:
                print("Invalid input, keeping current threshold")
    
    # Release webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change the threshold here (lower is stricter)
    setup_face_recognition_system(threshold=0.10)

# To use with test data:
# setup_face_recognition_system(use_test_data=True)

# To specify a specific pickle file:
# setup_face_recognition_system(pickle_path="FaceRec_Trained_Model.pickle")
