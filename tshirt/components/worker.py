import os
import attr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# tshirt
import lib
import lib.opencv


@attr.s
class Shirt(object):
    folder = attr.ib(default=None)
    imgsz = attr.ib(default=None) 
    pdfpage = attr.ib(default=None)
    h5db = attr.ib(default=None)
    idx_names = attr.ib(init=False)
    column_names = attr.ib(init=False)
    res_df = attr.ib(init=False)
    mopencv = attr.ib(init=False)
    backg = attr.ib(init=False)
    obj = attr.ib(init=False)
    msk = attr.ib(init=False)
    img = attr.ib(init=False)
    edges = attr.ib(init=False)
    msk_img = attr.ib(init=False)
    lines = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.idx_names = [u'name', u'name_id', u'line']
        self.column_names = ['x0','y0','theta', 'rho', 'deltax', 'deltay']
        idx = pd.MultiIndex(levels=[[]]*3, codes=[[]]*3, names=self.idx_names)
        self.res_df = pd.DataFrame(index=idx, columns=self.column_names)
        self.mopencv = lib.opencv

    def render(self, fid, fname):
        absfname = os.path.join(self.folder, fname)
        watershed = self.mopencv.monte_watershed_it(absfname)
        (self.backg, self.obj, self.msk, self.img) = watershed 
        self.edges, self.msk_img = self.mopencv.canny_it(self.msk)
        self.lines = self.mopencv.hough_it(self.edges)
        lib.helper.lines2df(fname, fid, self.lines, self.msk, self.imgsz, self.res_df)
        if self.pdfpage and not self.lines is None:
            self._graph(fname)
        if self.h5db and not self.lines is None:
            self._save2h5(fname)

    def get_res(self):
        return self.res_df.reset_index()[['name', 'rho', 'theta', 'x0', 'y0', 'deltax', 'deltay']]

    def _save2h5(self, fname):
        grp = self.h5db.create_group(fname)
        # #add some metadata to group
        # grp.attrs['name'] = 'main group'
        # grp.attrs['author'] = 'ksopyla'
        #create dataset in fname 
        img_ds = grp.create_dataset('img', data=self.img, compression="gzip")
        msk_ds = grp.create_dataset('msk', data=self.msk, compression="gzip")
        backg_ds = grp.create_dataset('backg', data=self.backg, compression="gzip")
        obj_ds = grp.create_dataset('obj', data=self.obj, compression="gzip")
        edges_ds = grp.create_dataset('edges', data=self.edges, compression="gzip")
        lines_ds = grp.create_dataset('lines', data=self.lines, compression="gzip")

    def _graph(self, fname):
        mplt = self.mopencv.plot_edge_detection
        fig, ax = mplt(self.img, self.msk, self.backg, self.obj, self.edges, self.lines)
        fig.suptitle(fname, fontsize=20)
        #plt.tight_layout()
        if type(self.pdfpage) is PdfPages:
            self.pdfpage.savefig(fig)
            plt.close(fig)
