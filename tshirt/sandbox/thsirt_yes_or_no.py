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
import h5py
import sys
import os
# tshirt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components import models

def printname(name):
    print(name)

def main():
    with  h5py.File('operation_t.h5','r') as hf:
        # show all objects in file
        # hf.visit(printname)
        for key in hf.keys():
            print()
            print(f"Get group: {key}.")
            grp = hf[key]
            #grp.visit(printname)
            for n, ds in grp.items():
                print(f"{n}: {ds}")


if __name__ == '__main__':
    RUNTIMEDIR = os.getcwd()
    main()