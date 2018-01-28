#!/home/cantab/miniconda3/envs/env2/bin/python2

import sys, time

import cv2
import numpy as np


filename = "Captured"
img = cv2.imread(filename + ".png")

if np.shape(img) == ():
    sys.exit("NO Image!!!")

m0,n0 = img.shape[0],img.shape[1]
img_target = np.zeros([m0,n0]).astype(np.uint8)

m1 = 0; m2 = m0
n1 = 0; n2 = n0
r = 5
while(True):
    cv2.imshow("Design the Target Image", img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    elif k == ord("1"):
        img = img[5:,:]
        m1 += 5

    elif k == ord("2"):
        img = img[:img.shape[0]-5,:]
        m2 -= 5

    elif k == ord("3"):
        img = img[:,5:]
        n1 += 5

    elif k == ord("4"):
        img = img[:,:img.shape[1]-5]
        n2 -= 5

    elif k == ord("c"):
        xx, yy = np.mgrid[:img.shape[0],:img.shape[1]]
        circle = (xx-int(img.shape[0]/2))**2 + (yy-int(img.shape[1]/2))**2
        r += 5
        img_filter = np.ones(img.shape).astype(np.uint8)
        rows,cols = np.where(circle<(r))
        img_filter[rows,cols,:] = 0
        img *= img_filter.astype(np.uint8)

    elif k == ord("s"):
        img = img[:,:,2]                                                        # Red image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                           # Gray image
        cv2.imwrite("Images/Target.png", img)
        print("m1,m2,n1,n2 = " + str([m1,m2,n1,n2]))
        print("m_target = " + str(img.shape[0]))
        print("n_target = " + str(img.shape[1]))
#         img_target[m1:m2, n1:n2] = img[:]
#         cv2.imwrite("Target Images/Target_" + filename  + "_FHD.png", img_target)
        break

cv2.destroyAllWindows()
