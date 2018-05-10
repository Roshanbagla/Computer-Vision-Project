"""Face detection and Face recognition."""
import os
import cv2
import numpy as np

subjects = ["", "Elvis Presley", "Roshan Wagle", "Ramiz"]


def load_classifierToDetectFace(image):
    """Loading a classifier."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    # checking the classifier load is successful or not
    test = face_cascade.load('lbpcascade_frontalface.xml')
    if test is True:
        find_face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1,
                                                  minNeighbors=5)
        print('Faces found: ', len(find_face))
        if len(find_face) > 0:
            for (x, y, w, h) in find_face:
                return gray_image[y:y+w, x:x+h], find_face[0]
    return None


# perfomring face recognition after detecting faces.
"""Face recognition using open cv."""


def training_data(folder_path):
    print(folder_path)
    """Return list of faces and labels."""
    directory = os.listdir(folder_path)
    print(directory)
    faces = []
    labels = []
    for directory_name in directory:
        if not directory_name.startswith("s"):
            continue
        label = int(directory_name.replace("s", ""))
        print("******************************")
        print("Reading directory ", label)
        subject_path = folder_path + "/" + directory_name
        subject_image_names = os.listdir(subject_path)
        for each_image_name in subject_image_names:
            if each_image_name.startswith("."):
                continue
            image_path = subject_path + "/" + each_image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image..", image)
            cv2.waitKey(100)
            face, rect = load_classifierToDetectFace(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                continue
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    return faces, labels


print("preparing data")
faces, labels = training_data("training")
print("data Prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

"""Create a Face Recognizer. """
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# train a face Recognizer
face_recognizer.train(faces, np.array(labels))


def rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def text_on_pic(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_image):
    copy_image = test_image.copy()
    face, rect = load_classifierToDetectFace(copy_image)
    label = face_recognizer.predict(face)
    text_label = subjects[label[0]]
    rectangle(copy_image, rect)
    text_on_pic(copy_image, text_label, rect[0], rect[1]-5)
    return copy_image


print(" predicting images")
test_image1 = cv2.imread("test-data/test1.jpg")
test_image2 = cv2.imread("test-data/test2.jpg")
test_image3 = cv2.imread("test-data/test3.jpg")


predicted_image1 = predict(test_image1)
predicted_image2 = predict(test_image2)
predicted_image3 = predict(test_image3)
print("prediction complete")

# display the images
cv2.imshow(subjects[1], predicted_image1)
cv2.imshow(subjects[2], predicted_image2)
cv2.imshow(subjects[3], predicted_image3)
cv2.waitKey(0)
cv2.destroyAllWindows()
