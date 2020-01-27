import sys
import numpy as np
import ROOT
import root_numpy as rnp
import uproot

visualization = True

tree = uproot.open('/eos/cms/store/cmst3/user/yiiyama/test_0.root')['events']
data = tree.arrays(['x', 'n', 'y'])

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
if visualization:
    vis3d = ROOT.TH3D('vis3d', '', 40, -1., 1., 40, -1., 1., 40, -1., 1.)
    vis3d.GetXaxis().SetTitle('Z')
    vis3d.GetYaxis().SetTitle('X')
    vis3d.GetZaxis().SetTitle('Y')
else:
    edist_ele = ROOT.TH1D('edist_ele', '', 100, 0., 1.)
    edist_pi = ROOT.TH1D('edist_pi', '', 100, 0., 1.)

for ievent, (event, nhit, gen_p) in enumerate(zip(data['x'], data['n'], data['y'])):
    if ievent != 50:
        continue
    
    if visualization:
        vis3d.Reset()
        
        for ihit in range(nhit):
            vis3d.Fill(event[ihit][2], event[ihit][0], event[ihit][1], event[ihit][3])
        
        #if gen_p == 0:
        #    part = 'electron'
        #else:
        #    part = 'pion'
        
        #vis3d.SetTitle(part)
        vis3d.Draw('BOX')
        
        canvas.Update()
        
        sys.stdin.readline()
    else:
        if gen_p == 0:
            rnp.fill_hist(edist_ele, event[:, 2], event[:, 5])
        else:
            rnp.fill_hist(edist_pi, event[:, 2], event[:, 5])

if not visualization:
    edist_ele.Scale(1. / edist_ele.GetEntries())
    edist_pi.Scale(1. / edist_pi.GetEntries())
    
    edist_ele.SetLineColor(ROOT.kRed)
    edist_pi.SetLineColor(ROOT.kBlue)
    
    edist_ele.Draw('HIST')
    edist_pi.Draw('HIST SAME')
    
    canvas.Update()
    
    sys.stdin.readline()
