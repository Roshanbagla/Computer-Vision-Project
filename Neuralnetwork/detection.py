"""Face detection using  Mulitasked Cascaded Convolutional Network."""

from mtcnn.mtcnn import MTCNN
import cv2
detector = MTCNN()
video = cv2.VideoCapture(0)
while True:
    __, frame = video.read()
    # Use mtcnn to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            # confidence = person['confidence']
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)
    # display resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
