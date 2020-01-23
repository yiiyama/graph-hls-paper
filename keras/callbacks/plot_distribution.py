import keras
import numpy as np
import ROOT
import root_numpy as rnp

class PlotDistribution(keras.callbacks.Callback):
    def __init__(self, data_path, input_type, n_vert_max, features=None, input_name='events'):
        super(PlotDistribution, self).__init__()

        if input_type == 'h5':
            import h5py
    
            data = h5py.File(data_path)
            self.x = data['x']
            self.n = data['n']
            y = data['y']
    
        elif input_type == 'root':
            import uproot
    
            data = uproot.open(data_path)[input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
            self.x = data['x']
            self.n = data['n']
            y = data['y']
            
        elif input_type == 'root-sparse':
            import uproot
            from generators.utils import to_dense
    
            data = uproot.open(data_path)[input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
            self.x = to_dense(data['n'], data['x'].content, n_vert_max=n_vert_max)
            self.n = data['n']
            y = data['y'][:, [0]]

        if features is not None:
            self.x = self.x[:, :, features]

        self.tarr = ROOT.TArrayD(y.shape[0])
        tcont = rnp.array(self.tarr, copy=False)
        tcont[:] = np.squeeze(y)

        self.parr = ROOT.TArrayD(y.shape[0])

        self.scatter = ROOT.TGraph(y.shape[0])
        self.scatter.SetMarkerSize(0.1)
        self.scatter.SetMarkerStyle(8)

        self.canvas = ROOT.TCanvas('plot_distribution', 'live', 600, 600)
        
    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict([self.x, self.n])
        pred = np.squeeze(pred)
        pcont = rnp.array(self.parr, copy=False)
        pcont[:] = np.squeeze(pred)
        #pcont *= 1024.

        self.canvas.Clear()
        self.scatter.DrawGraph(pred.shape[0], self.tarr.GetArray(), self.parr.GetArray(), 'AP')

        self.canvas.Update()
