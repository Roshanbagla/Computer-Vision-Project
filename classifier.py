"""Built a classifier to detect a face of human from an image."""

import cv2
MYIMAGE = cv2.imread('roshan.jpg', 0)           # read an image
cv2.imshow('image', MYIMAGE)
cv2.waitKey(0)
cv2.destroyAllWindows()
