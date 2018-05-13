# adapted from https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d
import cv2
import numpy as np 
import sys

if __name__ == '__main__':
    # read and scale down image from command line arg
    img = cv2.pyrDown(cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED))
    height, width = img.shape[:2]
    # blur convolution
    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
     
    # with each contour, draw boundingRect in green
    img_count = 0
    for c in contours: 
        x, y, w, h = cv2.boundingRect(c)
        # aim to only bound logic gates
        if w > 30 and w < 300 and h > 30 and h < 300:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = img[max(y - 50, 0):min(y + h + 50, height),
                        max(x - 50, 0):min(x + w + 50, width)]
            cv2.imwrite("roi" + str(img_count) + ".jpg", roi)
            img_count += 1
    
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
     
    cv2.imshow("contours", img)
     
    ESC = 27
    while True:
        keycode = cv2.waitKey()
        if keycode != -1:
            keycode &= 0xFF
            if keycode == ESC:
                break
    cv2.destroyAllWindows()