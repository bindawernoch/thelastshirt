import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_edge_detection(img, msk, backg, obj, edges, lines):
    if lines.any():
        for line in lines:
            for rho, theta in line:
                c = np.cos(theta)
                s = np.sin(theta)
                x0 = c*rho
                y0 = s*rho
                x1 = int(x0 + 5000*(-s))
                y1 = int(y0 + 5000*(c))
                x2 = int(x0 - 5000*(-s))
                y2 = int(y0 - 5000*(c))
                cv2.line(msk, (x1,y1), (x2,y2), (200,20,10), 20)

    fig1, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(14, 10.1), 
                                                      sharey='col', sharex='col')
    xlen = msk.shape[1]
    ylen = msk.shape[0]
    backg_x = [x*xlen for x in backg[1]]
    backg_y = [x*ylen for x in backg[0]]
    obj_x = [x*xlen for x in obj[1]]
    obj_y = [x*ylen for x in obj[0]]

    ax11.imshow(img)
    ax11.plot(backg_x, backg_y, 'o')
    ax11.plot(obj_x, obj_y, 'd')
    ax12.imshow(msk)
    ax12.plot(backg_x, backg_y, 'o')
    ax12.plot(obj_x, obj_y, 'd')
    ax21.imshow(edges, cmap='gray')
    ax22.imshow(msk)
    return fig1, (ax11, ax12, ax21, ax22)

def monte_watershed_it(img_f):
    # Open image
    img = cv2.imread(img_f, 3)
    assert img is not None, "Couldnt open {}".format(img_f)
    #
    mean = [0.6, 0.5]
    cov = [[0.01, 0], [0, 0.01]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 5).T
    if x.max() > 1:
        x /= x.max()
    if y.max() > 1:
        y /= y.max()
    #
    backg, obj = [[0.5, 0.5], [0.05, 0.95]], [y, x]
    #
    return watershed_it(img, backg, obj)

def hough_it(edges):
    #
    lines = cv2.HoughLines(edges, 1, np.pi/270, 100)  # , 140
    return lines

def canny_it(mtrx):
    mtrx_img = np.zeros(list(mtrx.shape)+[3])
    mtrx_img[:,:,1] = mtrx
    mtrx_img = np.array(mtrx_img, dtype = np.uint8)
    mtrx_img_grey = cv2.cvtColor(mtrx_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mtrx_img_grey, 100, 200) # , apertureSize = 5
    return edges, mtrx_img_grey

def sobel_it(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return sobelx+sobely

def watershed_it(img, backg, obj, plot_images=False):
    # https://stackoverflow.com/questions/42294109/remove-background-of-the-image-using-opencv-python

    # Figure setup
    if plot_images:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    if plot_images:
        ax1.imshow(img)

    # Remember image axis lengths
    ilen = img.shape[0]
    jlen = img.shape[1]

    # Create a blank image of zeros (same dimension as img)
    # It should be grayscale (1 color channel)
    marker = np.zeros_like(img[:,:,0]).astype(np.int32)

    # This step is manual. The goal is to find the points
    # which create the result we want. I suggest using a
    # tool to get the pixel coordinates.

    # Dictate the background and set the markers to 1
    for i, j in zip(*backg):
        marker[int(i*ilen)][int(j*jlen)] = 1

    # Dictate the area of interest
    # One might use different values for each part of the T-shirt
    for i, j in zip(*obj):
        marker[int(i*ilen)][int(j*jlen)] = 255
        marker[int(i*ilen)][int(j*jlen)] = 255

    # Now we have set the markers, we use the watershed
    # algorithm to generate a marked image
    marked = cv2.watershed(img, marker)

    # Plot this one. If it does what we want, proceed;
    # otherwise edit your markers and repeat
    if plot_images:
        ax2.imshow(marked, cmap='gray')

    # Make the background black, and what we want to keep white
    marked[marked <255] = 0
    # marked[marked > 1] = 255

    # Use a kernel to dilate the image, to not lose any detail on the outline
    # I used a kernel of 3x3 pixels
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations = 1)

    # Plot again to check whether the dilation is according to our needs
    # If not, repeat by using a smaller/bigger kernel, or more/less iterations
    if plot_images:
        ax3.imshow(dilation, cmap='gray')

    # Now apply the mask we created on the initial image
    final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))

    # cv2.imread reads the image as BGR, but matplotlib uses RGB
    # BGR to RGB so we can plot the image with accurate colors
    b, g, r = cv2.split(final_img)
    final_img = cv2.merge([r, g, b])

    # Plot the final result
    if plot_images:
        ax4.imshow(final_img)

    return backg, obj, dilation.astype(np.uint8), final_img
    