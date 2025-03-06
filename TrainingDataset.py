from imutils import paths
import face_recognition
import pickle
import cv2
import os


print("-----------------------------------------")
print("-------- Searching Datasets -------------")
print("-----------------------------------------")
imagePaths = list(paths.list_images("Dataset"))
print(imagePaths)

if(len(imagePaths) <= 0):
    print("-----------------------------------------")
    print("-----------No Datasets Found-------------")
    print("-----------------------------------------")
    print("Exiting.............")
else:
    print("We find data Successfully")

    

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print("We are Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)



    boxes = face_recognition.face_locations(image,
                                            model='cnn')  # cnn or RGB or hog model can be used.

    encodings = face_recognition.face_encodings(image, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

if(len(imagePaths) > 0):
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open('FaceRec_Trained_Model.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()
    




