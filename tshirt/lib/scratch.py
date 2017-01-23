import sys
import os
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2

def run_thresholds(img_f):

    img = cv2.imread(img_f)
    rows, cols, dummy = img.shape
    mrot = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
    dst = cv2.warpAffine(img, mrot, (cols, rows))

    ti0 = t.time()
    cv_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    ti1 = t.time()
    dt_cv = ti1-ti0
    print("Conversion took %0.5f seconds" % dt_cv)
    myfig, myax = plt.subplots() #figsize=(40, 20)
    myax.imshow(cv_rgb)

    plt.show()

run_thresholds("/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG")
