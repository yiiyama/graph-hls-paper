import sys
import numpy as np
import ROOT
import root_numpy as rnp
import uproot

geom_tree = uproot.open('geom.root')['detector']
geom_data = geom_tree.arrays(['x', 'y', 'z'])

coords = np.stack((geom_data['x'], geom_data['y'], geom_data['z']), axis=-1)

tree = uproot.open('/tmp/yiiyama/test_0.root')['events']
data = tree.arrays(['x', 'n', 'y'])

print data['x']

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
hist = ROOT.TH3D('hist', '', 40, 0., 1., 40, -1., 1., 40, -1., 1.)
hist.GetXaxis().SetTitle('Z')
hist.GetYaxis().SetTitle('X')
hist.GetZaxis().SetTitle('Y')

for event, nhit, gen_p in zip(data['x'], data['n'], data['y']):
    hist.Reset()

    for ihit in range(nhit):
        hist.Fill(event[ihit][2], event[ihit][0], event[ihit][1], event[ihit][5])

    if gen_p == 0:
        part = 'electron'
    else:
        part = 'pion'

    hist.SetTitle(part)
    hist.Draw('BOX')

    canvas.Update()

    sys.stdin.readline()
