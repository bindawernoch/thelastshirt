import io
import sys
import pathlib
import warnings
import numpy as np
from PIL import Image
from rembg.bg import remove
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# tshirt
import tshirt.lib
import tshirt.components


def operation_t(opts):
    mytsf = BASE / "tshirt_data/"
    exts = [".png"]  # ".jpg",
    for fh in [x for x in mytsf.glob("*") if x.suffix.lower() in exts]:
        with fh.open() as f:
            p = np.fromfile(f)
            result = remove(p)
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.savefig(fh.stem + "_out" + fh.suffix)


if __name__ == "__main__":
    CWD = pathlib.Path().cwd()
    BASE = pathlib.Path(__file__).parent
    MY_HOME = pathlib.Path().home()
    DATA = CWD
    OPTIONS = tshirt.lib.Options()
    operation_t(OPTIONS.parse(sys.argv[1:]))
