import os
import sys
import numpy as np
import h5py
from PIL import Image
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.image as mimage

plot_name = sys.argv[1]

fig = plt.figure(figsize=(12.8, 6.4), dpi=200)

with h5py.File('/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure2/prediction_keras_continuous.h5', 'r') as source:
    truth_c = source['truth_classification'][:]
    truth_r = source['truth_regression'][:]
    keras_continuous_c = source['pred_classification'][:]
    keras_continuous_r = source['pred_regression'][:]

with h5py.File('/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure2/prediction_keras_quantized.h5', 'r') as source:
    keras_quantized_c = source['pred_classification'][:]
    keras_quantized_r = source['pred_regression'][:]

with open('/afs/cern.ch/user/y/yiiyama/src/graph-hls-paper/hls4ml/combined_cpu2/prequantization/tb_data/csim_results.log') as source:
    hls_continuous_c = np.empty(keras_continuous_c.shape, dtype=np.float)
    hls_continuous_r = np.empty(keras_continuous_r.shape, dtype=np.float)
    for iline, line in enumerate(source):
        if iline % 2 == 0:
            hls_continuous_r[iline // 2] = float(line.strip())
        else:
            hls_continuous_c[iline // 2] = float(line.strip())

with open('/afs/cern.ch/user/y/yiiyama/src/graph-hls-paper/hls4ml/combined_cpu2/quantized/tb_data/csim_results.log') as source:
    hls_quantized_c = np.empty(keras_quantized_c.shape, dtype=np.float)
    hls_quantized_r = np.empty(keras_quantized_r.shape, dtype=np.float)
    for iline, line in enumerate(source):
        if iline % 2 == 0:
            hls_quantized_r[iline // 2] = float(line.strip())
        else:
            hls_quantized_c[iline // 2] = float(line.strip())

with h5py.File('/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure2/cut_based_roc.h5', 'r') as source:
    tpr_reference = source['tpr'][:]
    fpr_reference = source['fpr'][:]
    indices = np.argsort(tpr_reference)
    tpr_reference = tpr_reference[indices]
    fpr_reference = fpr_reference[indices]

with h5py.File('/afs/cern.ch/work/y/yiiyama/graph-hls-paper/figure2/prediction_weight_based.h5', 'r') as source:
    reference_r = source['prediction'][:]

logo = Image.open('hls4ml_logo.jpg')
size = np.array(logo.size).astype(np.float)
size = np.array(size * (0.05 * fig.bbox.ymax / size[1])).astype(np.int)
logo.thumbnail(tuple(size), Image.ANTIALIAS)

fpr_keras_continuous, tpr_keras_continuous, _ = roc_curve(truth_c, keras_continuous_c)
fpr_hls_continuous, tpr_hls_continuous, _ = roc_curve(truth_c, hls_continuous_c)
fpr_keras_quantized, tpr_keras_quantized, _ = roc_curve(truth_c, keras_quantized_c)
fpr_hls_quantized, tpr_hls_quantized, _ = roc_curve(truth_c, hls_quantized_c)

ebins = np.arange(10., 110., 10., dtype=np.float)
resp_keras_continuous = [[], []]
resp_keras_quantized = [[], []]
resp_hls_continuous = [[], []]
resp_hls_quantized = [[], []]
resp_reference = [[], []]

for ibin in range(ebins.shape[0] - 1):
    elow, ehigh = ebins[ibin:ibin + 2]
    indices = [np.asarray((truth_c == i) & (truth_r > elow * 1.e-2) & (truth_r < ehigh * 1.e-2)).nonzero() for i in (0, 1)]

    truth = [truth_r[idx] for idx in indices]
    keras_continuous = [keras_continuous_r[idx] for idx in indices]
    keras_quantized = [keras_quantized_r[idx] for idx in indices]
    hls_continuous = [hls_continuous_r[idx] for idx in indices]
    hls_quantized = [hls_quantized_r[idx] for idx in indices]
    reference = [reference_r[idx] for idx in indices]

    for i in (0, 1):
        resp_keras_continuous[i].append(keras_continuous[i] / truth[i])
        resp_keras_quantized[i].append(keras_quantized[i] / truth[i])
        resp_hls_continuous[i].append(hls_continuous[i] / truth[i])
        resp_hls_quantized[i].append(hls_quantized[i] / truth[i])
        resp_reference[i].append(reference[i] / truth[i])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

gs = gridspec.GridSpec(13, 2, hspace=0., wspace=0.25, figure=fig)

## Classification

ax = fig.add_subplot(gs[0, 0], frame_on=False, xticks=(), yticks=())
ax.text(0.5, 0.5, 'Classification', transform=ax.transAxes, va='center', ha='center', fontsize='xx-large')

ax = fig.add_subplot(gs[1:, 0])
plt.plot(1. - fpr_keras_continuous, tpr_keras_continuous, label='Keras continuous', color=colors[0])
plt.plot(1. - fpr_hls_continuous, tpr_hls_continuous, label='HLS continuous', color=colors[1], linestyle='--')
plt.plot(1. - fpr_keras_quantized, tpr_keras_quantized, label='Keras quantized', color=colors[2])
plt.plot(1. - fpr_hls_quantized, tpr_hls_quantized, label='HLS quantized', color=colors[3], linestyle='--')
plt.plot(1. - fpr_reference, tpr_reference, label='Cut-based', color=colors[4])
plt.xlim([0.7, 1.])
plt.ylim([0.7, 1.])
plt.xlabel('Pion rejection efficiency', fontsize='large')
plt.ylabel('Electron identification efficiency', fontsize='large')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, fontsize='large')
ax.tick_params(axis='both', labelsize='large')

axins = ax.inset_axes([0.12, 0.37, 0.35, 0.35])
axins.plot(1. - fpr_keras_continuous, tpr_keras_continuous, color=colors[0])
axins.plot(1. - fpr_hls_continuous, tpr_hls_continuous, color=colors[1], linestyle='--')
axins.plot(1. - fpr_keras_quantized, tpr_keras_quantized, color=colors[2])
axins.plot(1. - fpr_hls_quantized, tpr_hls_quantized, color=colors[3], linestyle='--')
axins.plot(1. - fpr_reference, tpr_reference, color=colors[4])
axins.set_xlim([0.9, 0.96])
axins.set_ylim([0.9, 0.96])
axins.tick_params(axis='both', labelsize='large')

logoins = ax.inset_axes([0.78, 0.925, 0.25, 0.07], frame_on=False)
logoins.tick_params(which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
logoins.imshow(logo)

## Regression

ax = fig.add_subplot(gs[0, 1], frame_on=False, xticks=(), yticks=())
ax.text(0.5, 0.5, 'Regression', transform=ax.transAxes, va='center', ha='center', fontsize='xx-large')

for ipart in (0, 1):
    if ipart == 0:
        ax0 = fig.add_subplot(gs[1:7, 1])
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.text(80., 0.52, 'Electrons', fontsize='large')
        plt.ylim([0.47, 2.47])
        ax0.tick_params(axis='y', labelsize='large')
    else:
        ax1 = fig.add_subplot(gs[7:, 1], sharex=ax0)
        plt.text(80., 0.3, 'Pions', fontsize='large')
        plt.ylim([0.25, 2.25])
        plt.xlabel('Primary particle energy (GeV)', fontsize='large')
        ax1.tick_params(axis='both', labelsize='large')

    plt.plot([ebins[0], ebins[-1]], [1., 1.], lw=1., ls='-', color='black')

    for ibox, data in enumerate([resp_keras_continuous, resp_hls_continuous, resp_keras_quantized, resp_hls_quantized, resp_reference]):
        positions = ebins[:-1] + 2.5 * (1 + ibox // 2)
        capprops = {'lw': 1., 'color': colors[ibox]}
        lstyle = dict(capprops)
        if ibox % 2 == 1:
            lstyle['ls'] = '--'

        plt.boxplot(data[ipart], whis=(5., 95.), sym='', positions=positions, manage_ticks=False, widths=2., whiskerprops=lstyle, boxprops=lstyle, capprops=capprops, medianprops=lstyle)

    plt.xlim([ebins[0], ebins[-1]])

handles = [mlines.Line2D([], [], color=colors[i], lw=1., ls=('--' if i % 2 == 1 else '-')) for i in range(4)]
handles.append(mlines.Line2D([], [], color=colors[4], lw=1.))
labels = ['Keras continuous', 'HLS continuous', 'Keras quantized', 'HLS quantized', 'Weight-based']
ax0.legend(handles, labels, fontsize='large', loc='upper right')

logoins = ax1.inset_axes([0.65, 0.85, 0.5, 0.14], frame_on=False)
logoins.tick_params(which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
logoins.imshow(logo)
    
fig.text(0.5, 0.45, 'Response', rotation='vertical', fontsize='large')

fig.savefig('%s.pdf' % plot_name, bbox_inches='tight')
fig.savefig('%s.png' % plot_name, bbox_inches='tight')
