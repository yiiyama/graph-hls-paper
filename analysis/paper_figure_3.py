import os
import sys
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox

plot_name = sys.argv[1]

sources = [
    ['/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure1/electron_events_2_2482.h5', '/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure1/pion_events_0_2419.h5'],
    ['/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure1/electron_dpde_2_2482.h5', '/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure1/pion_dpde_0_2419.h5']
]

try:
    os.makedirs(os.path.dirname(plot_name))
except OSError:
    pass

fig = plt.figure(figsize=(12.8, 9.6), dpi=200)

grow = 11
gcol = 16
half_grow = grow // 2
half_gcol = gcol // 2 - 1
gs = gridspec.GridSpec(grow, gcol, hspace=0.2, wspace=0.04, figure=fig)

for ititle in range(2):
    with h5py.File(sources[0][ititle], 'r') as source:
        prim = source['prim'][:]
        pu = source['pu'][:]
        pid = source['pid'][0]

    if pid == 0:
        part = 'Electron'
    else:
        part = 'Pion'

    title1 = '%s %.1f (%.1f) GeV' % (part, prim[0], prim[1])
    title2 = 'Pileup %.1f (%.1f) GeV' % (pu[0], pu[1])

    ax = fig.add_subplot(gs[0, 1 + ititle * half_gcol:1 + (ititle + 1) * half_gcol], frame_on=False, xticks=(), yticks=())
    ax.text(0.5, 0.8, title1, transform=ax.transAxes, va='center', ha='center', fontsize='xx-large')
    ax.text(0.5, 0.2, title2, transform=ax.transAxes, va='center', ha='center', fontsize='xx-large')

for irow in range(2):
    if irow == 0:
        row_label = '(a)'
    else:
        row_label = '(b)'

    ax = fig.add_subplot(gs[1 + irow * half_grow:1 + (irow + 1) * half_grow, 0], frame_on=False, xticks=(), yticks=())
    #ax.text(0.5, 0.5, row_label, transform=ax.transAxes, va='center', ha='center', fontsize='large')

    if irow == 0:
        # event display
        colors = {'red': ((0., 0., 0.), (1., 1., 1.)), 'green': ((0., 0., 0.), (1., 0., 0.)), 'blue': ((0., 1., 1.), (1., 0., 0.))}
        cmap_in = mpl.colors.LinearSegmentedColormap('rb', colors)
        cmap_out = cmap_in
    else:
        # dpde
        cmap_in = get_cmap('winter')
        cmap_out = get_cmap('Greys').reversed()

    axes = []

    for icol in range(2):
        ax = fig.add_subplot(gs[1 + irow * half_grow:1 + (irow + 1) * half_grow, 1 + icol * half_gcol:1 + (icol + 1) * half_gcol], projection='3d')
        ax.set_ylim(15., -15.)
        ax.set_xlabel('z (cm)', fontsize='x-large')
        ax.set_ylabel('x (cm)', fontsize='x-large')
        ax.set_zlabel('y (cm)', fontsize='x-large')
        axes.append(ax)

        with h5py.File(sources[irow][icol], 'r') as source:
            x_in = source['x_in'][:]
            y_in = source['y_in'][:]
            z_in = source['z_in'][:]
            s_in = source['s_in'][:]
            c_in = source['c_in'][:]
            x_out = source['x_out'][:]
            y_out = source['y_out'][:]
            z_out = source['z_out'][:]
            s_out = source['s_out'][:]
            if irow == 0:
                c_out = source['c_out'][:]

        if irow == 0:
            vmin_in, vmax_in = 0., 1.
        else:
            c_out = np.ones_like(s_out) * 0.2
            vmin_in, vmax_in = -0.5, 1.2
    
        vmin_out, vmax_out = 0., 1.
    
        p_in = ax.scatter(xs=x_in, ys=z_in, zs=y_in, c=c_in, s=s_in, alpha=1., vmin=vmin_in, vmax=vmax_in, cmap=cmap_in)
        ax.scatter(xs=x_out, ys=z_out, zs=y_out, c=c_out, s=s_out, alpha=0.3, vmin=vmin_out, vmax=vmax_out, cmap=cmap_out)

        x_loc = mticker.FixedLocator([-15., 0., 15.])
        y_loc = mticker.FixedLocator([0., 15., 30., 45.])
        z_loc = mticker.FixedLocator([-15., 0., 15.])
        ax.xaxis.set_major_locator(y_loc)
        ax.yaxis.set_major_locator(z_loc)
        ax.zaxis.set_major_locator(x_loc)

        ax.tick_params('x', labelsize='x-large')
        ax.tick_params('y', labelsize='x-large')
        ax.tick_params('z', labelsize='x-large')

    ax = fig.add_subplot(gs[1 + irow * half_grow:1 + (irow + 1) * half_grow, -1], frame_on=False, xticks=(), yticks=(), clip_on=False)

    cbar = fig.colorbar(p_in, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize='x-large')
    if irow == 0:
        cbar.set_label('Primary fraction', fontsize='x-large')
    else:
        cbar.set_label(r'$\Delta E_{\mathrm{pred}}/\Delta h$', fontsize='x-large')

handles = [
    mlines.Line2D([], [], alpha=1., color='black', marker='o', linestyle='None', markersize=10),
    mlines.Line2D([], [], alpha=0.3, color='black', marker='o', linestyle='None', markersize=10)
]
labels = ['Clustered', 'Unclustered']
fig.legend(handles, labels, loc='upper right', frameon=False, bbox_to_anchor=(0.86, 0.4), fontsize='x-large')

fig.savefig('%s.pdf' % plot_name, bbox_inches='tight', pad_inches=0.2)
fig.savefig('%s.png' % plot_name, bbox_inches='tight', pad_inches=0.2)
