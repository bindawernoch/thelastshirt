
import os
import cv2
import scipy
import itertools
import time as t
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# from IPython import embed
# embed() # drop into an IPython session.
#         # Any variables you define or modify here
#         # will not affect program execution

# from IPython.core.debugger import Pdb
# Pdb().set_trace()


def run_thresholds(img_f):
    """ development stuff """

    # import sys
    # print(sys.path)
    # plt.ion()

    # open image
    img = cv2.imread(img_f)
    # rotate image (i guess i shouldn't do that)
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
    mypp = PdfPages('/home/mario/Dropbox/Tshirts/scratch/adaptive_thresh42.pdf')

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



def run_watershed(img_f, backg, obj, plot_images=False):
    # https://stackoverflow.com/questions/42294109/remove-background-of-the-image-using-opencv-python

    # Figure setup
    if plot_images:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    # Load the image
    img = cv2.imread(img_f, 3)
    assert img is not None, "Couldnt open {}".format(img_f)

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

        # print(i*ilen, j*jlen)
    # marker[1500][250] = 1
    # marker[1500][-250] = 1


    # Dictate the area of interest
    # One might use different values for each part of the T-shirt
    for i, j in zip(*obj):
        marker[int(i*ilen)][int(j*jlen)] = 255
        marker[int(i*ilen)][int(j*jlen)] = 255

        # print(i*ilen, j*jlen)
    # marker[1500][2500] = 255    # ...
    # marker[135][294] = 64     # ...

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

    return dilation.astype(np.uint8), final_img

def iris_flowers():
    # https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    # Load libraries
    import pandas
    from pandas.tools.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

    # shape
    print(dataset.shape)
    # head
    print(dataset.head(20))
    # describe
    print(dataset.describe())
    # class distribution
    print(dataset.groupby('class').size())

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    # histograms
    dataset.hist()
    # scatter plot matrix
    scatter_matrix(dataset)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    # Test options and evaluation metric
    scoring = 'accuracy'
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def sobel(img_f):

    img = cv2.imread(img_f,0)

    # timg = cv2.threshold(img, 0.5)
    # img = timg

    plt.subplot(111),plt.imshow(img)


    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    plt.figure()
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    plt.figure()
    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

    # https://de.wikipedia.org/wiki/Hough-Transformation


def test_gauss():
    mean = [0.6, 0.5]
    cov = [[0.01, 0], [0, 0.01]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 5).T
    fig2, ax21 = plt.subplots(1, 1, )
    ax21.plot(x, y, 'o')
    print(x)
    print(y)
    plt.axis('equal')

def get_contour_lines(img):

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    return sobelx+sobely

if __name__ == '__main__':
    # first attempt
    # tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG"
    # run_thresholds(tst)
    # run_watershed(tst, backg, obj)

    # second attempt
    # tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/"
    # [[fraction of y axis], [fraction of x axis]]
    # backg, obj = [[0.5, 0.5], [0.05, 0.95]], [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]]
    #
    # for fn in os.listdir(tst):
    #     fnap = os.path.join(tst, fn)
    #     if os.path.isfile(fnap):
    #         run_watershed(fnap, backg, obj)

    # third attempt
    # iris_flowers()
    # https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/

    # fourth
    # from IPython.core import debugger
    # debug = debugger.Pdb().set_trace
    # debug()
    # tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG"
    # sobel(tst)

    ## fifth
    tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000711.JPG"
    mean = [0.6, 0.5]
    cov = [[0.01, 0], [0, 0.01]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 5).T
    #
    backg, obj = [[0.5, 0.5], [0.05, 0.95]], [y, x]
    msk, res = run_watershed(tst, backg, obj)
    #
    cntrs = get_contour_lines(msk)
    #
    msk_im = np.zeros(list(msk.shape)+[3])
    msk_im[:,:,1] = msk
    msk_im = np.array(msk_im, dtype = np.uint8)
    msk_im_grey = cv2.cvtColor(msk_im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(msk_im_grey, 0, 1000, apertureSize = 5)

    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(edges,1, np.pi/180, 100, minLineLength, maxLineGap)
    # for x1,y1,x2,y2 in lines[0]:
    #     print(x1,y1,x2,y2)
    #     cv2.line(msk_im,(x1,y1),(x2,y2),(255,0,0),20)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
    for line in lines:
        for rho, theta in line:
            c = np.cos(theta)
            s = np.sin(theta)
            x0 = c*rho
            y0 = s*rho
            print()
            print(x0, y0)
            print(90 - np.rad2deg(theta))

            x1 = int(x0 + 5000*(-s))
            y1 = int(y0 + 5000*(c))
            x2 = int(x0 - 5000*(-s))
            y2 = int(y0 - 5000*(c))

            cv2.line(msk_im,(x1,y1),(x2,y2),(255,0,0),20)


    fig1, (ax11, ax12, ax13) = plt.subplots(1, 3, sharey='col', figsize=(20,6))
    xlen = msk.shape[1]
    ylen = msk.shape[0]
    backg_x = [x*xlen for x in backg[1]]
    backg_y = [x*ylen for x in backg[0]]
    obj_x = [x*xlen for x in obj[1]]
    obj_y = [x*ylen for x in obj[0]]

    ax11.plot(backg_x, backg_y, 'o')
    ax11.plot(obj_x, obj_y, 'd')
    ax11.imshow(msk, origin='low')
    ax12.imshow(cntrs, origin='low')
    ax13.imshow(msk_im, origin='low')


    plt.show()
