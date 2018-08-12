import numpy as np


def lines2df(row_name, n_id, lines, detres_df):
    if lines.any():
        for i, line in enumerate(lines):
            for rho, theta in line:
                c = np.cos(theta)
                s = np.sin(theta)
                x0 = c*rho
                y0 = s*rho
                detres_df.loc[(row_name, n_id, i), 'x0'] = c*rho
                detres_df.loc[(row_name, n_id, i), 'y0'] = s*rho
                detres_df.loc[(row_name, n_id, i), 'theta'] = 90 - np.rad2deg(theta)
                detres_df.loc[(row_name, n_id, i), 'rho'] = rho
