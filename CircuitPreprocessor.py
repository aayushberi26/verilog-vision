import cv2
import numpy as np
import sys

class CircuitPreprocessor:
    '''
    Adapted from https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d
    '''
    def __init__(self, img_path):
        # read and scale down image from command line arg
        self.img = cv2.pyrDown(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        self.height, self.width = self.img.shape[:2]
        
        # blur convolution
        kernel = np.ones((3,3),np.float32)/9
        self.img = cv2.filter2D(self.img,-1,kernel)
        self.contours = self.get_contours()
        self.bounding_rects = self.get_bounding_rects()

    def get_contours(self):
        # threshold image
        ret, threshed_img = cv2.threshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),
                        127, 255, cv2.THRESH_BINARY)
        # find contours and get the external one
        image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_bounding_rects(self):
        rects = []
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            # constrain to only bound logic gate sized contours
            if w > 30 and w < 275 and h > 30 and h < 275:
                hull = cv2.convexHull(contour)
                epsilon = 0.1*cv2.arcLength(contour,True)
                hull_approx = cv2.approxPolyDP(hull, epsilon, True)
                hull_area = cv2.contourArea(hull_approx)
                print (x, y, w, h)
                print (hull_area)
                rects.append((x,y,w,h))
        return rects

    def view_bounding_rects(self):
        for x,y,w,h in self.bounding_rects:
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(self.img, self.contours, -1, (255, 255, 0), 1)        
        cv2.imshow("contours", self.img)

        # display until kill signal given via ESC key
        keycode = cv2.waitKey()
        cv2.destroyAllWindows()  

    def extract_gates(self):
        img_count = 0
        for x,y,w,h in self.bounding_rects:
            roi = self.img[max(y - 50, 0):min(y + h + 50, self.height),
                        max(x - 50, 0):min(x + w + 50, self.width)]
            cv2.imwrite("temp/roi" + str(img_count) + ".jpg", roi)
            img_count += 1

if __name__ == '__main__':
    cp = CircuitPreprocessor(sys.argv[1])
    cp.extract_gates()
    cp.view_bounding_rects()