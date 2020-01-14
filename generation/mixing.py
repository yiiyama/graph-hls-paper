import os
import sys
import glob
import time
import tempfile
import shutil
import numpy as np
import uproot
from argparse import ArgumentParser

arg_parser = ArgumentParser(description='Run simple jobs on condor')
arg_parser.add_argument('--dataset', '-s', metavar='TYPE', dest='dataset_type', default='classification', help='Task type of the dataset (classification, clustering, regression(?)).')
arg_parser.add_argument('--format', '-f', metavar='FORMAT', dest='output_format', default='root', help='Output file format (h5, root, root-sparse).')
arg_parser.add_argument('--nevt', '-n', metavar='N', dest='nevt', default=1000, type=int, help='Number of events in one file.')
arg_parser.add_argument('--nfile', '-m', metavar='N', dest='nfile', default=1, type=int, help='Number of files to produce.')
arg_parser.add_argument('--first-file', '-i', metavar='N', dest='first_ifile', default=0, type=int, help='Index of the first output file.')
arg_parser.add_argument('--out', '-o', metavar='PATH', dest='outname', default='mixing', help='Output file name without the serial number and extension.')
arg_parser.add_argument('--write-processed', '-x', metavar='PATH', dest='write_processed', help='Write the paths of processed files to PATH.')
arg_parser.add_argument('--skip-processed', '-y', metavar='PATH', dest='skip_processed', help='Skip input files listed in PATH.')
arg_parser.add_argument('--add-pu', '-P', action='store_true', dest='add_pu', help='Add pileup.')

args = arg_parser.parse_args()
del sys.argv[1:]

if args.output_format == 'h5':
    import h5py
elif args.output_format == 'root':
    import root_numpy as rnp
elif args.output_format == 'root-sparse':
    import ROOT

with tempfile.NamedTemporaryFile(delete=False) as t:
    tmpname = t.name

try:
    os.makedirs(os.path.dirname(args.outname))
except:
    pass

geom_data = uproot.open('geom.root')['detector'].arrays(['id', 'x', 'y', 'z', 'dxy', 'dz'])
# bugfix
geom_data['dz'] = np.where(geom_data['z'] < 350. + 0.6 * 25, 0.6, 4.2)

ele_paths = glob.glob('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated/electron_10_100/*/events_*.root')
pi_paths = glob.glob('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated/pioncharged_10_100/*/events_*.root')
if args.add_pu:
    pu_paths = glob.glob('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated/pileup_0_0/*/events_*.root')

if args.write_processed:
    processed_paths = open(args.write_processed, 'w')

if args.skip_processed:
    with open(args.skip_processed) as source:
        skip_paths = set(line.strip() for line in source)

    pathss = [ele_paths, pi_paths]
    if args.add_pu:
        pathss.append(pu_paths)

    for paths in pathss:
        paths_tmp = []
        for path in paths:
            if path not in skip_paths:
                paths_tmp.append(path)
    
        del paths[:]
        paths.extend(paths_tmp)

processed_paths_list = []
def make_generator(paths, branches):
    def get_event():
        current_path = ''
        for path, data in uproot.iterate(paths, 'events', branches, reportpath=True):
            if path != current_path:
                print 'Opened file', path
                processed_paths_list.append(path)

            for ievt in range(data[branches[0]].shape[0]):
                yield tuple(data[b][ievt] for b in branches)

    return get_event

if args.dataset_type == 'classification':
    cluster_size_max = 256
    cluster_radius = 6.4

    coords = np.stack((geom_data['x'], geom_data['y']), axis=-1)
    geom_stack = np.stack((
            geom_data['x'] / 18.,
            geom_data['y'] / 18.,
            (geom_data['z'] - 350.) / 120.,
            #geom_data['dxy'] / 6.,
            #geom_data['dz'] / 4.2
    ), axis=-1)

    electrons = make_generator(ele_paths, ['recoEnergy'])()
    pions = make_generator(pi_paths, ['recoEnergy'])()
    if args.add_pu:
        pus = make_generator(pu_paths, ['recoEnergy'])()

    time_read = 0.
    time_process = 0.
    time_write = 0.

    def make_cluster(ipart):
        global time_read
        global time_process

        t = time.time()

        if ipart == 0:
            prim, = next(electrons)
        else:
            prim, = next(pions)

        if args.add_pu:
            pu, = next(pus)

        time_read += time.time() - t

        t = time.time()

        if args.add_pu:
            event = prim + pu
        else:
            event = prim

        iseed = np.argmax(event)

        seed_axis = np.tile(coords[iseed], (coords.shape[0], 1))
        dr2 = np.sum(np.square(coords - seed_axis), axis=1)
        in_radius = np.asarray((dr2 < cluster_radius ** 2) & (event > 100.)).nonzero()

        cluster_tmp = np.concatenate((
            geom_stack,
            np.expand_dims(np.sqrt(event * 1.e-3), axis=-1)
        ), axis=-1)[in_radius]

        time_process += time.time() - t

        return cluster_tmp

    ifile = 0
    while ifile != args.nfile:
        try:
            if args.output_format == 'h5':
                out_x = np.empty((args.nevt, cluster_size_max, 6), dtype=np.float32)
                out_n = np.empty((args.nevt,), dtype=np.int16)
                out_y = np.empty((args.nevt,), dtype=np.int8)
    
            elif args.output_format == 'root':
                out_x = np.zeros((cluster_size_max, 6), dtype=np.float32)
                out_entries = np.empty((args.nevt,), dtype=[('x', np.float32, (cluster_size_max, 6)), ('n', np.int16), ('y', np.int8)])
    
            elif args.output_format == 'root-sparse':
                output = ROOT.TFile.Open(tmpname, 'recreate')
                out_tree = ROOT.TTree('events', '')
                out_x = np.empty((cluster_size_max, 6), dtype=np.float32)
                out_n = np.empty((1,), dtype=np.int32)
                out_y = np.empty((1,), dtype=np.int32)
                out_tree.Branch('n', out_n, 'n/I')
                out_tree.Branch('x', out_x, 'x[n][6]/F')
                out_tree.Branch('y', out_y, 'y/I')
    
            for iev, ipart in enumerate(np.random.randint(0, 2, args.nevt)):
                cluster_tmp = make_cluster(ipart)

                n = cluster_tmp.shape[0]
                if n > cluster_size_max:
                    print 'large cluster: ', cluster_tmp.shape[0]
                    n = cluster_size_max

                t = time.time()
        
                if args.output_format == 'h5':
                    out_x[iev, :n] = cluster_tmp[:n]
                    out_n[iev] = n
                    out_y[iev] = ipart
                elif args.output_format == 'root':
                    out_x[:n] = cluster_tmp[:n]
                    out_x[n:] *= 0.
                    out_entries[iev] = (out_x, n, ipart)
                elif args.output_format == 'root-sparse':
                    out_x[:n] = cluster_tmp[:n]
                    out_n[0] = n
                    out_y[0] = ipart
                    out_tree.Fill()

                time_write += time.time() - t

        except StopIteration:
            print 'Early stop due to input exhaustion.'
            ifile = args.nfile - 1

        if args.write_processed:
            for path in processed_paths_list:
                processed_paths.write(path + '\n')

            del processed_paths_list[:]

        t = time.time()

        if args.output_format == 'h5':
            with h5py.File(tmpname, 'w', libver='latest') as output:
                chunk_size = min(args.nevt, 1024)
                out_clusters = output.create_dataset('x', (args.nevt, cluster_size_max, 6), chunks=(chunk_size, cluster_size_max, 6), compression='gzip', dtype='f')
                out_size = output.create_dataset('n', (args.nevt,), chunks=(chunk_size,), compression='gzip', dtype='i')
                out_truth = output.create_dataset('y', (args.nevt,), chunks=(chunk_size,), compression='gzip', dtype='i')
                out_clusters.write_direct(out_x)
                out_size.write_direct(out_n)
                out_truth.write_direct(out_y)

            extension = 'h5'

        elif args.output_format == 'root':
            rnp.array2root(out_entries, tmpname, treename='events', mode='recreate')
            extension = 'root'

        elif args.output_format == 'root-sparse':
            output.cd()
            out_tree.Write()
            output.Close()
            extension = 'root'

        shutil.move(tmpname, '%s_%d.%s' % (args.outname, ifile + args.first_ifile, extension))

        time_write += time.time() - t

        ifile += 1

    print 'Read time:', time_read, 'Process time:', time_process, 'Write time:', time_write
