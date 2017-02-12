import time as t
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2


def run_thresholds(img_f):
    """ development stuff """

    # open image
    img = cv2.imread(img_f)
    # rotate image (i guess i shouldnt do that)
    rows, cols, dummy = img.shape
    mrot = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
    dst = cv2.warpAffine(img, mrot, (cols, rows))
    # convert imgae to rgb
    ti0 = t.time()
    cv_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    ti1 = t.time()
    dt_cv = ti1-ti0
    print("Conversion took %0.5f seconds" % dt_cv)
    # convert image to grayscale
    gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)

    # start experiment
    mypp = PdfPages('/home/mario/Dropbox/Tshirts/scratch/adaptive_thresh6.pdf')

    for i in range(401, 501, 2):
        th1 = cv2.adaptiveThreshold(gray, gray.max(), 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, i, 8)
        myfig, myax = plt.subplots()
        myax.imshow(th1)
        myax.set_title('{a}_{b}'.format(a=i, b=8))
        mypp.savefig(myfig)
        plt.close(myfig)
    mypp.close()


if __name__ == '__main__':
    run_thresholds("/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG")