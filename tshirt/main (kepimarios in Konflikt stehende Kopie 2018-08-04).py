import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
# tshirt
import tshirt.lib as lib
import tshirt.components.opencv as myopencv

def operation_r(opts):

    plt.show()

def single_t(img_n, detres_df, plotit=True):

    detres_df = pd.DataFrame(columns=['x0','y0','theta'])
    tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000903.JPG"
    #
    msk, final_img, backg, obj = myopencv.monte_watershed_it(tst)
    #
    edges, msk_img = myopencv.canny_it(msk)
    #
    lines = myopencv.hough_it(edges)
    #
    n = os.path.basename(tst)
    print(n)
    # lib.helper.lines2df( , lines, detres_df)
    if plotit:
        myopencv.plot_edge_detection(final_img, msk, backg, obj, msk_img, edges, lines)





if __name__ == '__main__':
    OPTIONS = lib.Options()
    single_t(OPTIONS.parse(sys.argv[1:]))
