import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import numpy as np
import kneed
import sys
import os
# tshirt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components import models


def main():
    mypp = PdfPages(os.path.join(RUNTIMEDIR, "kneedle_operation_t.pdf"))

    #  Tshirt Data
    df = pd.read_csv(os.path.join(RUNTIMEDIR, "temp.csv"))
    theta_min = df.theta.min()
    df['theta_nrm']  = df.theta
    df['theta_nrm'] -= theta_min
    theta_max = df.theta_nrm.max()
    rho_max = df.rho.max()
    df['theta_nrm'] /= theta_max
    df['rho_nrm'] = df.rho
    df['rho_nrm'] /= rho_max

    #  Regions
    rgns = alt.pd.DataFrame(models.get_classifications())
    print(rgns)
    alt_rgns = alt.Chart(rgns).mark_rect(opacity=0.2)
    regcolor = alt.Color('name:N', legend=None, scale=alt.Scale(scheme='set1'))
    plt_rect_rgns = alt_rgns.encode(x='xstart:Q', x2='xend:Q',
                                    y='ystart:Q', y2='yend:Q', color=regcolor)

    #  Kmeans ALL Tshirt Data
    all_dta, fig = kmeans_it(df.loc[:, ['rho_nrm', 'theta_nrm']].values)
    df.loc[:, 'id'] = all_dta
    #  Altair it
    allcolor = alt.Color('id:N', scale=alt.Scale(scheme='set1'))
    plt_kmeans = alt.Chart(df).mark_point().encode(x='rho:Q', y='theta:Q',
                                                   color=allcolor)
    (plt_kmeans + plt_rect_rgns).save('chart_kmeans.html')

    #  Kmeans each Tshirt Data
    alt_graphs = []
    for name, subdf in df.groupby('name', sort=False):
        isubdf = subdf.copy()
        kminp = isubdf.loc[:, ['rho_nrm', 'theta_nrm']].values
        single_dta, single_fig = kmeans_it(kminp)
        isubdf.loc[:, 'sub_id'] = single_dta
        #
        single_fig.suptitle(name)
        mypp.savefig(single_fig)
        plt.close(single_fig)
        #
        ialt = alt.Chart(isubdf).mark_point().properties(title=name)
        scol = alt.Color('sub_id:N', scale=alt.Scale(scheme='set1'))
        subplt_kmeans = ialt.encode(x='rho:Q',  y='theta:Q', color=scol)
        alt_graphs.append((subplt_kmeans + plt_rect_rgns))
    mypp.close()
    alt.vconcat(*alt_graphs).resolve_scale(color='independent').save('charts_kmeans.html')

def kmeans_it(X):
    if X.shape[0] > 12:
        n_clusters = 12
    else:
        return [1] * X.shape[0], plt.figure()
    my_ests = [KMeans(n_clusters=i) for i in range(1, n_clusters)]
    my_sods = np.array([None]*len(my_ests))  # SumOfDeltaSquares
    for i, my_est in enumerate(my_ests):
        my_est.fit(X)
        sods = 0
        for j, center in enumerate(my_est.cluster_centers_):
            my_indices = np.where(my_est.labels_ == j)[0]
            delta_vectors = X[my_indices, :] - center
            deltas = delta_vectors[:, 0]**2 + delta_vectors[:, 1]**2
            sods += deltas.sum()
        my_sods[i] = sods
    kneedle = kneed.KneeLocator(range(my_sods.size), my_sods,
                                curve='convex', direction='decreasing')
    kneedle_delta = 2
    ret_fig = plt.figure(figsize=(8, 8))
    plt.plot(range(1, my_sods.size + 1), my_sods, '-d')
    plt.vlines(kneedle.knee + 1 + kneedle_delta, plt.ylim()[0], plt.ylim()[1])
    ret_data = [x for x in list(my_ests[kneedle.knee + kneedle_delta].labels_)]
    ret_data = np.array(ret_data) + 1
    return ret_data, ret_fig


if __name__ == '__main__':
    RUNTIMEDIR = os.getcwd()
    main()
