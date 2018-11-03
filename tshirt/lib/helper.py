import numpy as np
import pprint as pp
from scipy import ndimage


def lines2df(row_name, row_id, lines, mask, msize, res_df):
    norm = [1, 1]
    for i, (a, b) in enumerate(zip(mask.shape, msize)):
        norm[i] = b / a
    middle = (mask.shape[0] / 2, mask.shape[1] / 2)
    com = ndimage.measurements.center_of_mass(mask)
    if not lines is None:
        if lines.any():
            for i, line in enumerate(lines):
                for rho, theta in line:
                    c = np.cos(theta)
                    s = np.sin(theta)
                    res_df.loc[(row_name, row_id, i), 'x0'] = int(c*rho * norm[1])
                    res_df.loc[(row_name, row_id, i), 'y0'] = int(s*rho * norm[0])
                    res_df.loc[(row_name, row_id, i), 'theta'] = int(90 - np.rad2deg(theta))
                    #res_df.loc[(row_name, row_id, i), 'rho'] = int(rho)
                    res_df.loc[(row_name, row_id, i), 'deltax'] = int((middle[1] - com[1] * norm[1]))
                    res_df.loc[(row_name, row_id, i), 'deltay'] = int((middle[0] - com[0] * norm[0]))
    res_df.rho = np.sqrt( (res_df.x0 + res_df.deltax)**2 + (res_df.y0 + res_df.deltay)**2 )
