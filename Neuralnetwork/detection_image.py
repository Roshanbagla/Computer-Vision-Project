from mtcnn.mtcnn import MTCNN
import cv2
detector = MTCNN()
# video = cv2.VideoCapture(0)
image = cv2.imread("low_light.jpg")
result = detector.detect_faces(image)
bounding_box = result[0]['box']
cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0, 155, 255),
              2)
cv2.imwrite("front.jpg", image)
cv2.namedWindow("Image")
cv2.imshow("image", image)
cv2.waitKey(0)
