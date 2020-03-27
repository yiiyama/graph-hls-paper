import sys
import numpy as np
import ROOT
import root_numpy as rnp
import uproot

visualization = False

tree = uproot.open('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/mixed/combined/events_0.root')['events']
data = tree.arrays(['x', 'n', 'y'])

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
if visualization:
    vis3d = ROOT.TH3D('vis3d', '', 40, 0., 1., 40, -1., 1., 40, -1., 1.)
    vis3d.GetXaxis().SetTitle('Z')
    vis3d.GetYaxis().SetTitle('X')
    vis3d.GetZaxis().SetTitle('Y')
else:
    edist_ele = ROOT.TH1D('edist_ele', '', 100, 0., 1.)
    edist_pi = ROOT.TH1D('edist_pi', '', 100, 0., 1.)
    resp_ele = ROOT.TH1D('resp_ele', '', 100, -1., 1.)
    resp_pi = ROOT.TH1D('resp_pi', '', 100, -1., 1.)

for ievent, (event, nhit, gen_p) in enumerate(zip(data['x'], data['n'], data['y'])):
    if visualization:
        vis3d.Reset()

        rnp.fill_hist(vis3d, event[:, [2, 0, 1]], event[:, 3])
        #for ihit in range(nhit):
        #    vis3d.Fill(event[ihit][2], event[ihit][0], event[ihit][1], event[ihit][3])
        
        if gen_p[0] == 0:
            part = 'electron'
        else:
            part = 'pion'
        
        vis3d.SetTitle(part)
        vis3d.Draw('BOX')
        
        canvas.Update()
        
        sys.stdin.readline()
    else:
        resp = (np.sum(event[:, 3], axis=0) - gen_p[1]) / gen_p[1]
        if gen_p[0] == 0:
            rnp.fill_hist(edist_ele, event[:, 2], event[:, 3])
            resp_ele.Fill(resp)
        else:
            rnp.fill_hist(edist_pi, event[:, 2], event[:, 3])
            resp_pi.Fill(resp)

if not visualization:
    plot_dir = '/tmp'

    edist_ele.Scale(1. / edist_ele.GetEntries())
    edist_pi.Scale(1. / edist_pi.GetEntries())
    
    edist_ele.SetLineColor(ROOT.kRed)
    edist_pi.SetLineColor(ROOT.kBlue)
    
    edist_ele.Draw('HIST')
    edist_pi.Draw('HIST SAME')
    
    canvas.Print(plot_dir + '/edist_pu_100.png')
    canvas.Print(plot_dir + '/edist_pu_100.pdf')

    resp_ele.Scale(1. / resp_ele.GetEntries())
    resp_pi.Scale(1. / resp_pi.GetEntries())

    resp_ele.SetLineColor(ROOT.kRed)
    resp_pi.SetLineColor(ROOT.kBlue)
    
    resp_ele.Draw('HIST')
    resp_pi.Draw('HIST SAME')

    canvas.Print(plot_dir + '/resp_pu_100.png')
    canvas.Print(plot_dir + '/resp_pu_100.pdf')    
