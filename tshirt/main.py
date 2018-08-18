import os
import sys
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# tshirt
import tshirt.lib as lib
import tshirt.components.opencv as myopencv

RUNTIMEDIR = os.getcwd()


def operation_t(opts):
    my_index = pd.MultiIndex(levels=[[]]*3, labels=[[]]*3,
                             names=[u'name', u'name_id', u'line'])
    detres_df = pd.DataFrame(index=my_index, columns=['x0','y0','theta'])

    mytsf = "/home/mario/Dropbox/Tshirts/tshirt_proj/data/"
    mypp = PdfPages(os.path.join(RUNTIMEDIR, "operation_t.pdf"))
    #
    for i, fn in enumerate(os.listdir(mytsf)):
        fnap = os.path.join(mytsf, fn)
        if os.path.isfile(fnap):
            single_t(fnap, i, detres_df , plotit=mypp)
        # if i == 5:
        #     break
    mypp.close()

    df = detres_df.reset_index()[['name', 'rho', 'theta']]
    df.rho = np.abs(df.rho)
    df.theta = np.abs(df.theta)
    alt.Chart(df).mark_point().encode(x='rho', y='theta', color='name').save('chart.html')
    #
    charts = [alt.Chart(subdf).mark_point().properties(title=name).encode(x='rho', y='theta')
              for name, subdf in df.groupby('name', sort=False)]
    alt.vconcat(*charts).save('charts.html')

def single_t(img_n, n_id, detres_df, plotit=False):
    msk, final_img, backg, obj = myopencv.monte_watershed_it(img_n)
    edges, msk_img = myopencv.canny_it(msk)
    lines = myopencv.hough_it(edges)
    #
    if not lines is None:
        image_just_name = os.path.basename(img_n)
        lib.helper.lines2df(image_just_name, n_id, lines, detres_df)
        #
        if plotit:
            fig, ax = myopencv.plot_edge_detection(final_img, msk, backg, obj, 
                                                   msk_img, edges, lines)
            fig.suptitle(image_just_name, fontsize=20)
            #plt.tight_layout()
            if type(plotit) is PdfPages:
                plotit.savefig(fig)
                plt.close(fig)

if __name__ == '__main__':
    OPTIONS = lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
