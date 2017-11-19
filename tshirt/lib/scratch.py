
import time as t

import cv2 as cvtwo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def run_thresholds(img_f):
    """ development stuff """

    # import sys
    # print(sys.path)
    plt.ion()

    # open image
    img = cvtwo.imread(img_f)
    # rotate image (i guess i shouldn't do that)
    rows, cols, dummy = img.shape
    mrot = cvtwo.getRotationMatrix2D((cols/2, rows/2), -90, 1)
    dst = cvtwo.warpAffine(img, mrot, (cols, rows))
    # convert imgae to rgb
    ti0 = t.time()
    cv_rgb = cvtwo.cvtColor(dst, cvtwo.COLOR_BGR2RGB)
    ti1 = t.time()
    dt_cv = ti1-ti0
    print("Conversion took %0.5f seconds" % dt_cv)
    # convert image to grayscale
    gray = cvtwo.cvtColor(cv_rgb, cvtwo.COLOR_BGR2GRAY)

    # start experiment
    mypp = PdfPages('/home/mario/Dropbox/Tshirts/scratch/adaptive_thresh42.pdf')

    for i in range(401, 501, 2):
        th1 = cvtwo.adaptiveThreshold(gray, gray.max(), 
                                    cvtwo.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cvtwo.THRESH_BINARY, i, 8)
        myfig, myax = plt.subplots()
        myax.imshow(th1)
        myax.set_title('{a}_{b}'.format(a=i, b=8))
        mypp.savefig(myfig)
        plt.close(myfig)
    mypp.close()



def run_watershed(img_f):
    # https://stackoverflow.com/questions/42294109/remove-background-of-the-image-using-opencv-python

    f1, f2 = plt.figure(), plt.figure()
    af1 = f1.add_subplot(111)
    af2 = f2.add_subplot(111)

    # Load the image
    img = cvtwo.imread(img_f, 3)

    # show input
    af1.imshow(img)

    # Create a blank image of zeros (same dimension as img)
    # It should be grayscale (1 color channel)
    marker = np.zeros_like(img[:,:,0]).astype(np.int32)

    # This step is manual. The goal is to find the points
    # which create the result we want. I suggest using a
    # tool to get the pixel coordinates.

    # Dictate the background and set the markers to 1
    marker[1500][250] = 1
    marker[1500][-250] = 1


    # Dictate the area of interest
    # I used different values for each part of the T-shirt
    marker[1500][2500] = 255    # car body
    marker[135][294] = 64     # rooftop


    # # rear bumper
    # marker[225][456] = 128
    # marker[224][461] = 128
    # marker[216][461] = 128

    # # front wheel
    # marker[225][189] = 192
    # marker[240][147] = 192

    # # rear wheel
    # marker[258][409] = 192
    # marker[257][391] = 192
    # marker[254][421] = 192

    # Now we have set the markers, we use the watershed
    # algorithm to generate a marked image
    marked = cvtwo.watershed(img, marker)

    # Plot this one. If it does what we want, proceed;
    # otherwise edit your markers and repeat
    af2.imshow(marked, cmap='gray')

    return 0

    # Make the background black, and what we want to keep white
    marked[marked == 1] = 0
    marked[marked > 1] = 255

    # Use a kernel to dilate the image, to not lose any detail on the outline
    # I used a kernel of 3x3 pixels
    kernel = np.ones((3,3),np.uint8)
    dilation = cvtwo.dilate(marked.astype(np.float32), kernel, iterations = 1)

    # Plot again to check whether the dilation is according to our needs
    # If not, repeat by using a smaller/bigger kernel, or more/less iterations
    plt.imshow(dilation, cmap='gray')
    plt.show()

    # Now apply the mask we created on the initial image
    final_img = cvtwo.bitwise_and(img, img, mask=dilation.astype(np.uint8))

    # cvtwo.imread reads the image as BGR, but matplotlib uses RGB
    # BGR to RGB so we can plot the image with accurate colors
    b, g, r = cvtwo.split(final_img)
    final_img = cvtwo.merge([r, g, b])

    # Plot the final result
    plt.imshow(final_img)
    plt.show()

if __name__ == '__main__':
    tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG"
    # run_thresholds(tst)
    run_watershed(tst)

    plt.show()