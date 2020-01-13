import sys
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
import root_numpy as rnp
import uproot

geom_tree = uproot.open('geom.root')['detector']
geom_data = geom_tree.arrays(['x', 'y', 'z'])

coords = np.stack((geom_data['x'], geom_data['y'], geom_data['z']), axis=-1)

data_blocks = uproot.iterate('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated/123456/electron_10_100/events_22.root', 'events', ['genPid', 'genEnergy', 'genX', 'genY', 'recoEnergy'], entrysteps=128)

hcont = ROOT.TH2D('containment', '', 101, 0., 1.01, 55, 4., 15.)

for idata, data_block in enumerate(data_blocks):
    for ient, (gen_p, gen_e, gen_x, gen_y, reco_e) in enumerate(zip(data_block['genPid'], data_block['genEnergy'], data_block['genX'], data_block['genY'], data_block['recoEnergy'])):
        if gen_e == 0.:
            continue
        
        iseed = np.argmax(reco_e)
        xseed, yseed = coords[iseed, 0:2]
        seed = np.tile(np.array([xseed, yseed]), (coords.shape[0], 1))
        dr2 = np.sum(np.square(coords[:, 0:2] - seed), axis=1)

        for r in np.arange(4., 15., 0.2, dtype=np.float):
            in_radius = np.asarray(dr2 < r * r).nonzero()
            total = np.sum(reco_e[in_radius]) * 1.e-3
            containment = total / gen_e

            hcont.Fill(containment, r)

#    if idata == 10:
#        break

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
hcont.Draw('COLZ')
canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/200110_cylinder/electron.pdf')
canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/200110_cylinder/electron.png')
