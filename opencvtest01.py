# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 04:22:20 2015

@author: chiwakii
"""

# -*- coding: utf-8 -*-
 
import cv2
import numpy as np
 
if __name__ == '__main__':
 
    cap = cv2.VideoCapture(0)
 
    ret, flame = cap.read()
 
    # グレースケール変換
    img_gray_current = cv2.cvtColor(flame, cv2.COLOR_BGR2GRAY)
 
    # HSV変換
    img_hsv = cv2.cvtColor(flame, cv2.COLOR_BGR2HSV)
 
 
    while(True):
 
        ret, flame = cap.read()
 
        img_gray_next = cv2.cvtColor(flame, cv2.COLOR_BGR2GRAY)
 
        flow = cv2.calcOpticalFlowFarneback(img_gray_current,
                                            img_gray_next,
                                            None,
                                            0.5,
                                            3,
                                            15,
                                            3,
                                            5,
                                            1.2,
                                            0)
        img_gray_current = img_gray_next
 
        magnitude, angle = cv2.cartToPolar(flow[...,0],
                                           flow[...,1])
        img_hsv[...,0] = angle * 180 / np.pi / 2
        img_hsv[...,2] = cv2.normalize(magnitude,
                                   None,
                                   0,
                                   255,
                                   cv2.NORM_MINMAX)
 
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
 
        cv2.imshow("test", img_bgr)
 
 
        # qを押したら終了。
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
