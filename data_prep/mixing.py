#!/usr/bin/env python

import os
import sys
argv = list(sys.argv)
del sys.argv[1:]
import glob
import shutil
import tempfile
import uproot
import ROOT
import numpy as np

from argparse import ArgumentParser

arg_parser = ArgumentParser(description='Run simple jobs on condor')
arg_parser.add_argument('--nevt', '-n', metavar='N', dest='nevt', default=1000, type=int, help='Number of events in one file.')
arg_parser.add_argument('--nfile', '-m', metavar='N', dest='nfile', default=1, type=int, help='Number of files to produce.')
arg_parser.add_argument('--first-file', '-i', metavar='N', dest='first_ifile', default=0, type=int, help='Index of the first output file.')
arg_parser.add_argument('--source', '-c', metavar='PATH', dest='source', default='/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated', help='Source directory of generated events file')
arg_parser.add_argument('--out', '-o', metavar='PATH', dest='outname', default='mixing', help='Output file name without the serial number and extension.')
arg_parser.add_argument('--write-processed', '-x', metavar='PATH', dest='write_processed', help='Write the paths of processed files to PATH.')
arg_parser.add_argument('--skip-processed', '-y', metavar='PATH', dest='skip_processed', help='Skip input files listed in PATH.')

sys.argv = argv

args = arg_parser.parse_args()

with tempfile.NamedTemporaryFile(delete=False) as t:
    tmpname = t.name

cluster_size_max = 256
nfeat = 4

processed_paths_list = []
def make_generator(paths, branches, report_ievt=False):
    def get_event():
        current_path = ''
        for path, data in uproot.iterate(paths, 'clusters', branches, reportpath=True):
            if path != current_path:
                print 'Opened file', path
                processed_paths_list.append(path)

            for ievt in range(data[branches[0]].shape[0]):
                if report_ievt:
                    print path, ievt
                yield tuple(data[b][ievt] for b in branches)

    return get_event


ele_paths = glob.glob('%s/SingleE_*/ntuple_*.root' % args.source)
pi_paths = glob.glob('%s/SinglePion_*/ntuple_*.root' % args.source)

electrons = make_generator(ele_paths, ['x', 'n', 'y', 'gen_pt'])()
pions = make_generator(pi_paths, ['x', 'n', 'y', 'gen_pt'])()

ifile = 0
while ifile != args.nfile:
    output = ROOT.TFile.Open(tmpname, 'recreate')
    out_tree = ROOT.TTree('events', '')
    out_x = np.empty((cluster_size_max, nfeat), dtype=np.float32)
    out_n = np.empty((1,), dtype=np.int32)
    out_y = np.empty((1,), dtype=np.int32)
    out_tree.Branch('n', out_n, 'n/I')
    out_tree.Branch('x', out_x, 'x[n][%d]/F' % nfeat)
    out_tree.Branch('y', out_y, 'y/I')

    try:
        for iev, ipart in enumerate(np.random.randint(0, 2, args.nevt)):
            if ipart == 0:
                x, n, y, pt = next(electrons)
            else:
                x, n, y, pt = next(pions)

            if pt < 20.:
                continue

            out_x[:n] = x[:n]
            out_n[0] = n
            out_y[0] = y
            out_tree.Fill()

    except StopIteration:
        print 'Early stop due to input exhaustion.'
        break

    finally:
        output.cd()
        out_tree.Write()
        output.Close()
        extension = 'root'
    
        shutil.move(tmpname, '%s_%d.%s' % (args.outname, ifile + args.first_ifile, extension))
    
    ifile += 1
