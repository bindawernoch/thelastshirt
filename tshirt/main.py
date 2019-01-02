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
        # if i == 3:
        #     break
    mypp.close()
    df = mshirt.get_res()
    # regions
    rgns = alt.pd.DataFrame(components.models.get_classifications())
    rect_rgns = alt.Chart(rgns).mark_rect().encode(x='xstart:Q', x2='xend:Q',
                                                   y='ystart:Q', y2='yend:Q',
                                                   color='name:N')
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
    alt.vconcat(*charts).save('charts.html')
    #
    df.to_csv('temp.csv')


if __name__ == '__main__':
    RUNTIMEDIR = os.getcwd()
    OPTIONS = lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
