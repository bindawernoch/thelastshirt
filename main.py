import os
import sys
import h5py
import socket
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# tshirt
import tshirt.lib
import tshirt.components

from rembg.bg import remove
from PIL import Image

def operation_t(opts):
    mytsf = BASE / "tshirt_data/"

    for i, fn in enumerate(os.listdir(mytsf)):
        if os.path.isfile(os.path.join(mytsf, fn)):
            f = np.fromfile(fn)
            result = remove(f)
            img = Image.open(io.BytesIO(result)).convert("RGBA")
            print(fn)
        if i == 2:
            break

    


if __name__ == "__main__":
    CWD = pathlib.Path().cwd()
    BASE = pathlib.Path(__file__).parent
    MY_HOME = pathlib.Path().home()
    DATA = CWD
    OPTIONS = tshirt.lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
