import sys
import numpy as np
import ROOT
import root_numpy as rnp
import uproot

geom_tree = uproot.open('geom50.root')['detector']
geom_data = geom_tree.arrays(['x', 'y', 'z'])

coords = np.stack((geom_data['x'], geom_data['y'], geom_data['z']), axis=-1)

data_blocks = uproot.iterate('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated/pioncharged_10_100/65432/events_1.root', 'events', ['genPid', 'genEnergy', 'genX', 'genY', 'recoEnergy'], entrysteps=128)

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
hist = ROOT.TH3D('hist', '', 40, 350., 350. + 0.6 * 25 + 4.2 * 25, 40, -30., 30., 40, -30., 30.)
hist.GetXaxis().SetTitle('Z')
hist.GetYaxis().SetTitle('X')
hist.GetZaxis().SetTitle('Y')

for data_block in data_blocks:
    for ient, (gen_p, gen_e, gen_x, gen_y) in enumerate(zip(data_block['genPid'], data_block['genEnergy'], data_block['genX'], data_block['genY'])):
        hist.Reset()
    
        for hit, (x, y, z) in zip(data_block['recoEnergy'][ient], coords):
            hist.Fill(z, x, y, hit)
    
        if gen_p == 0:
            part = 'electron'
        else:
            part = 'pion'
    
        total = np.sum(data_block['recoEnergy'][ient]) * 1.e-3
    
        hist.SetTitle('%s, %.1f GeV, (%.1f, %.1f) -> %.1f GeV' % (part, gen_e, gen_x, gen_y, total))
        hist.Draw('BOX')
    
        canvas.Update()
    
        sys.stdin.readline()
