import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_fn = './haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_fn)

    rects = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects):
        rects[:,2:] += rects[:,:2]
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,123), 2)
            cv2.imshow('camera', frame)
    else:
        # Display the resulting frame
        cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
