import sys
import os
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
from matplotlib.backends.backend_pdf import PdfPages

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

    # myfig, myax = plt.subplots() #figsize=(40, 20)
    # myax.imshow(cv_rgb)
    # plt.show()

    gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)

    th1 = cv2.adaptiveThreshold(gray, gray.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 3, 1)

    # myfig, myax = plt.subplots()
    # myax.imshow(th1)
    # plt.show()


    mypp = PdfPages('/home/mario/Dropbox/Tshirts/scratch/adaptive_thresh.pdf')

    for i in range(3, 17, 2):
        for j in range(1, 20):
            th1 = cv2.adaptiveThreshold(gray, gray.max(), 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, i, j)
            myfig, myax = plt.subplots()
            myax.imshow(th1)
            myax.set_title('{a}_{b}'.format(a=i, b=j))
            mypp.savefig(myfig)
            plt.close(myfig)
    mypp.close()




run_thresholds("/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG")


#pp = PdfPages('/home/mario/Dropbox/Tshirts/scratch/thresh/multipage.pdf')
#
#for thresh_perc in range(101):
#    im_bw = cv2.threshold(gray, gray.max()*thresh_perc/100., gray.max(), cv2.THRESH_BINARY)[1]
#    #im_bw_inv = cv2.bitwise_not(im_bw)
#    fig, ax = plt.subplots(figsize=(40, 20))
#    ax.imshow(im_bw);
#    pp.savefig(fig)
#    plt.close(fig)
#pp.close()