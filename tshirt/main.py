import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
# tshirt
import tshirt.lib as lib
import tshirt.components.opencv as myopencv

def operation_t(opts):
    my_index = pd.MultiIndex(levels=[[]]*3, labels=[[]]*3,
                             names=[u'name', u'name_id', u'line'])
    detres_df = pd.DataFrame(index=my_index, columns=['x0','y0','theta'])

    # tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/P1000903.JPG"
    # single_t(tst, detres_df)

    tst = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/"
    for i, fn in enumerate(os.listdir(tst)):
        fnap = os.path.join(tst, fn)
        if os.path.isfile(fnap):
            single_t(fnap, i, detres_df)  # , plotit=False
        if i == 4:
            break

    # mdf = pd.pivot_table(detres_df.reset_index(), columns=['name'], index=['theta', 'rho'])
    fig, ax = plt.subplots()
    detres_df.rho = detres_df.rho.apply(abs)
    detres_df.theta = detres_df.theta.apply(abs)
    mdf = detres_df.reset_index()
    mdf.plot(x='rho', y='theta', c='name_id', ax=ax, kind='scatter', legend=True,
             grid = True, subplots=True, colormap='viridis', s=mdf['name_id']*50, 
             alpha=0.5)
    plt.show()

def single_t(img_n, n_id, detres_df, plotit=True):
    msk, final_img, backg, obj = myopencv.monte_watershed_it(img_n)
    edges, msk_img = myopencv.canny_it(msk)
    lines = myopencv.hough_it(edges)
    #
    if not lines is None:
        lib.helper.lines2df(os.path.basename(img_n), n_id, lines, detres_df)
        #
        if plotit:
            myopencv.plot_edge_detection(final_img, msk, backg, obj, 
                                        msk_img, edges, lines)
    

if __name__ == '__main__':
    OPTIONS = lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
