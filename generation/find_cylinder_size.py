import sys
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
import root_numpy as rnp
import uproot

plots_dir = '/tmp'

geom_tree = uproot.open('geom.root')['detector']
geom_data = geom_tree.arrays(['x', 'y', 'z'])

coords = np.stack((geom_data['x'], geom_data['y'], geom_data['z']), axis=-1)

#part = 'pioncharged'
part = 'electron'

data_blocks = uproot.iterate('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated2/%s_10_100/123456/events_22.root' % part, 'events', ['genPid', 'genEnergy', 'genX', 'genY', 'recoEnergy'], entrysteps=128)

matrix = ROOT.TH2D('matrix', '', 100, 0., 1., 55, 4., 15.)
nent = 0

for idata, data_block in enumerate(data_blocks):
    for gen_p, gen_e, gen_x, gen_y, reco_e in zip(*tuple(data_block[k] for k in ['genPid', 'genEnergy', 'genX', 'genY', 'recoEnergy'])):
        if gen_e == 0.:
            continue
        
        iseed = np.argmax(reco_e)
        xseed, yseed = coords[iseed, 0:2]
        seed = np.tile(np.array([xseed, yseed]), (coords.shape[0], 1))
        dr2 = np.sum(np.square(coords[:, 0:2] - seed), axis=1)

        c = 0.
        for r in np.arange(4., 15., 0.2, dtype=np.float):
            in_radius = np.asarray(dr2 < r * r).nonzero()
            total = np.sum(reco_e[in_radius]) * 1.e-3
            containment = total / gen_e

            while c < containment:
                matrix.Fill(c, r)
                c += 0.01

            if c >= 1.:
                break

        nent += 1

arr = rnp.hist2array(matrix)
quantile = np.cumsum(arr, axis=1) / nent

graph = ROOT.TGraph(100)

for ip in range(100):
    q = np.searchsorted(quantile[ip], 0.95)
    graph.SetPoint(ip, ip * 0.01, matrix.GetYaxis().GetBinCenter(q + 1))

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
graph.Draw('APL')
canvas.Print(plots_dir + '/%s_200cm.pdf' % part)
canvas.Print(plots_dir + '/%s_200cm.png' % part)
