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
import lib
import components


def operation_t(opts):
    mytsf = "/home/mario/Dropbox/Tshirts/tshirt_data/"
    mypp = PdfPages(os.path.join(RUNTIMEDIR, "operation_t.pdf"))
    mysze = (3000, 4000)
    mshirt = components.worker.Shirt(mytsf, mysze, mypp) 
    #
    for i, fn in enumerate(os.listdir(mytsf)):
        if os.path.isfile(os.path.join(mytsf, fn)):
            mshirt.render(i, fn)
        # if i == 10:
        #     break
    mypp.close()
    df = mshirt.get_res()
    # !!!!! CONSIDER SIGN INFORMATION / dont drop it / add it to graph for both rho and theta
    # df.rho = np.sqrt( (df.x0 + df.deltax)**2 + (df.y0 + df.deltay)**2 )
    # df.theta = np.abs(df.theta)
    # regions
    rgns = [{
            "xstart": 1200,
            "xend": 1500,
            "ystart": 10,
            "yend": 14,
            "event": "Waist Left Side"
            },
            {
            "xstart": 800,
            "xend": 1200,
            "ystart": -1,
            "yend": 12,
            "event": "Waist Left Side"
            },
            {
            "xstart": 400,
            "xend": 800,
            "ystart": -7,
            "yend": 9,
            "event": "Waist Left Side"
            },
                        {
            "xstart": 250,
            "xend": 400,
            "ystart": -7,
            "yend": 8,
            "event": "Waist Left Side"
            },
            {
            "xstart": 0,
            "xend": 250,
            "ystart": -14,
            "yend": 2,
            "event": "Sleeve Left End"
            },
            {
            "xstart": 250,
            "xend": 500,
            "ystart": -26,
            "yend": -7,
            "event": "Sleeve Left End"
            },
            {
            "xstart": 500,
            "xend": 750,
            "ystart": -32,
            "yend": -7,
            "event": "Sleeve Left End"
            },
            {
            "xstart": 600,
            "xend": 850,
            "ystart": -39,
            "yend": -25,
            "event": "Sleeve Left End"
            },
            {
            "xstart": 2600,
            "xend": 2900,
            "ystart": 0,
            "yend": 8,
            "event": "Waist Right Side"
            },
            {
            "xstart": 2300,
            "xend": 2600,
            "ystart": -10,
            "yend": 6,
            "event": "Waist Right Side"
            },
            {
            "xstart": 2050,
            "xend": 2300,
            "ystart": -12,
            "yend": 3,
            "event": "Waist Right Side"
            },
            {
            "xstart": 1800,
            "xend": 2050,
            "ystart": -13,
            "yend": 0,
            "event": "Waist Right Side"
            },
            {
            "xstart": 1500,
            "xend": 1800,
            "ystart": -16,
            "yend": -3,
            "event": "Waist Right Side"
            },
            {
            "xstart": 2850,
            "xend": 3300,
            "ystart": -1,
            "yend": 31,
            "event": "Sleeve Right End"
            }]
    rgns = alt.pd.DataFrame(rgns)
    rect_rgns = alt.Chart(rgns).mark_rect().encode(x='xstart:Q', x2='xend:Q',
                                                   y='ystart:Q', y2='yend:Q',
                                                   color='event:N')
    #
    pnt_single = alt.Chart(df).mark_point()
    # single
    pnt_single.encode(x='rho:Q', y='theta:Q', color='name:N').save('chart.html')
    # multiple
    domx = [0, max(mysze)]
    domy = [-95, 95]
    charts = [rect_rgns + 
              alt.Chart(subdf).mark_point(color='#333')
                             .properties(title=name)
                             .encode(alt.X('rho:Q', scale=alt.Scale(domain=domx)),
                                     alt.Y('theta:Q', scale=alt.Scale(domain=domy)))
              for name, subdf in df.groupby('name', sort=False)]
    #
    alt.vconcat(*charts).save('charts.html')

if __name__ == '__main__':
    RUNTIMEDIR = os.getcwd()
    OPTIONS = lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
