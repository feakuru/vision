import cv2
import numpy as np
import imutils
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./face_predictor.dat')

old_shapes = []
counter = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if counter % 4 == 0:
        old_shapes = []
        rects = detector(gray, 1)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            frame = face_utils.visualize_facial_landmarks(frame, shape)
            old_shapes.append(shape)
    else:
        for shape in old_shapes:
            frame = face_utils.visualize_facial_landmarks(frame, shape)
    cv2.imshow("camera", frame)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
