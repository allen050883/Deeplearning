import cv2
import pytesseract

import numpy as np

NETWORK_W = 608
NETWORK_H = 608

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, NETWORK_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NETWORK_H)

#%%

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    #cv2.imwrite('mask.png', mask)
    
    contours, _ = cv2.findContours(mask.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(c) for c in contours]
    
    if len(area) != 0:
        j = np.argmax(area)
        print("Contour ", j, "enclose the largest area.")
        print("The area is ", area[j])
        
        img_read_1 = cv2.drawContours(frame.copy(), contours, j, (0, 255, 0), 2)
        
        cnt = contours[j]
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        app_cnt = [approx]
        img_1 = cv2.drawContours(frame.copy(), app_cnt, 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        
        catch = frame[y:int(y+h/2), x:x+w,:]
        
        try:
            text = pytesseract.image_to_string(catch, config='--psm 11')
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        except ValueError:
            pass
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
