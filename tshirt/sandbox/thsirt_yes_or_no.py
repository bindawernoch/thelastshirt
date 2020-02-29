import warnings

warnings.filterwarnings("ignore")
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import numpy as np
import pathlib
import socket
import kneed
import h5py
import sys
import os

# tshirt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components import models


def printname(name):
    print(name)


def main():
    operaton_t = DATA / "operation_t.h5"
    with h5py.File(str(operaton_t), "r") as hf:
        # show all objects in file
        # hf.visit(printname)
        for key in hf.keys():
            print()
            # print(f"Get group: {key}.")
            print("Get group: {key}.".format(key=key))
            grp = hf[key]
            # grp.visit(printname)
            for n, ds in grp.items():
                # print(f"{n}: {ds}")
                print("{n}: {ds}".format(n=n, ds=ds))


if __name__ == "__main__":
    CWD = pathlib.Path().cwd()
    MY_HOME = pathlib.Path().home()
    if socket.gethostname() == "penguin":
        DATA = MY_HOME / "data"
    else:
        DATA = CWD
    main()
