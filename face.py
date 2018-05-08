"""open cv face detection implemenation using Haar Classifer."""
import cv2
PATH = 'thor.jpg'


def grayscale(image_path):
    """Convert the image to gray color."""
    MYIMAGE = cv2.imread(image_path)           # read an image
    return MYIMAGE


def load_classifierToDetectFace(gray_image):
    """Loading a classifier."""
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # checking the classifier load is successful or not
    test = face_cascade.load('haarcascade_frontalface_default.xml')
    if test is True:
        find_face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2,
                                                  minNeighbors=5)
        print('Faces found: ', len(find_face))
        if len(find_face) > 0:
            for (x, y, w, h) in find_face:
                cv2.rectangle(gray_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return gray_image
    return None


gray_image = grayscale(PATH)
image_with_face = load_classifierToDetectFace(gray_image)
if image_with_face is not None:
    cv2.imshow('image', image_with_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
