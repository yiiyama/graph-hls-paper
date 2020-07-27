from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
try:
    import tensorflow.keras as keras
except ImportError:
    import keras

from generators.utils import to_dense

class UprootJaggedSequence(keras.utils.Sequence):
    def __init__(self, paths, batch_size, num_steps=None, format='xn', features=None, n_vert_restrict=None, y_dtype=np.int, y_features=None, sample_weighting=None, dataset_name='events'):
        super(UprootJaggedSequence, self).__init__()
        
        self.batch_size = batch_size
        self.n_vert_restrict = n_vert_restrict

        self.format = format

        n_events = 0
        self.n_vert_max = 1
        for data in uproot.iterate(paths, dataset_name, ['n'], namedecode='ascii'):
            n_events += data['n'].shape[0]
            try:
                max_n = np.amax(data['n'])
            except ValueError:
                max_n = 0
            while self.n_vert_max < max_n: # this is not a great way to find the max size; better to explicitly write it down in the file as metadata
                self.n_vert_max *= 2

        print('n_vert_max =', self.n_vert_max)

        self.n_steps = n_events // batch_size

        if num_steps is not None and num_steps < self.n_steps:
            self.n_steps = num_steps
            n_events = num_steps * batch_size
    
        self.n = np.empty((n_events,), dtype=np.int)

        shape = (n_events,)
        if y_features is None or type(y_features) is int:
            self.y = np.empty(shape, dtype=y_dtype)
        elif type(y_features) is dict:
            self.y = dict((key, np.empty(shape, dtype=y_dtype[key])) for key in y_features)
        elif type(y_features) is list:
            self.y = [np.empty(shape, dtype=y_dtype[i]) for i in range(len(y_features))]
    
        start = 0
        for data in uproot.iterate(paths, dataset_name, ['n', 'y'], namedecode='ascii'):
            end = start + data['n'].shape[0]
            if end > n_events:
                end = n_events

            nread = end - start

            # cannot restrict n here to n_vert_restrict because self.n is used to convert jagged -> dense in __getitem__
            self.n[start:end] = data['n'][:nread]

            if y_features is None:
                self.y[start:end] = data['y'][:nread]
            elif type(y_features) is int:
                self.y[start:end] = data['y'][:nread, y_features]
            elif type(y_features) is dict:
                for key, idx in y_features.items():
                    self.y[key][start:end] = data['y'][:nread, idx]
            elif type(y_features) is list:
                for i, idx in enumerate(y_features):
                    self.y[i][start:end] = data['y'][:nread, idx]

            if end == n_events:
                break

            start = end

        if features is None:
            for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii', entrysteps=1):
                nfeat = data['x'].content.shape[1]
                break
        else:
            nfeat = len(features)
    
        nhits_total = np.sum(self.n)
        self.xcont = np.empty((nhits_total, nfeat), dtype=np.float)
    
        cstart = 0
        for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii'):
            x = data['x'].content
            cend = cstart + x.shape[0]
            if cend > self.xcont.shape[0]:
                cend = self.xcont.shape[0]

            nread = cend - cstart

            if features is None:
                self.xcont[cstart:cend] = x[:nread]
            else:
                self.xcont[cstart:cend] = x[:nread, features]

            if cend == self.xcont.shape[0]:
                break
    
            cstart = cend

        self.xpos = np.empty((self.n_steps + 1,), dtype=np.int)
        xpos = 0
        for index in range(self.n_steps):
            self.xpos[index] = xpos
            start = self.batch_size * index
            end = start + self.batch_size
            xpos += np.sum(self.n[start:end])

        self.xpos[self.n_steps] = xpos

        self.sample_weighting = sample_weighting

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        nbatch = self.n[start:end]

        xstart = self.xpos[index]
        xend = self.xpos[index + 1]
        x = to_dense(nbatch, self.xcont[xstart:xend], n_vert_max=self.n_vert_max)[:, :self.n_vert_restrict] # [:None] has no effect

        if self.n_vert_restrict is not None:
            nbatch = np.minimum(nbatch, np.ones(nbatch.shape, dtype=np.int) * self.n_vert_restrict)

        if type(self.y) is dict:
            y = dict((key, value[start:end]) for key, value in self.y.items())
        elif type(self.y) is list:
            y = [value[start:end] for value in self.y]
        else:
            y = self.y[start:end]

        if self.sample_weighting is None:
            if self.format == 'xn':
                return [x, nbatch], y
            elif self.format == 'xen':
                return [x[:, :, :3], x[:, :, 3], nbatch], y
            elif self.format == 'x':
                return x, y
        else:
            w = self.sample_weighting(x, y)
            if self.format == 'xn':
                return [x, nbatch], y, w
            elif self.format == 'xen':
                return [x[:, :, :3], x[:, :, 3], nbatch], y, w
            elif self.format == 'x':
                return x, y, w

        
def make_generator(paths, batch_size, num_steps=None, format='xn', features=None, n_vert_restrict=None, y_dtype=np.int, y_features=None, sample_weighting=None, dataset_name='events'):
    sequence = UprootJaggedSequence(paths, batch_size, num_steps, format, features, n_vert_restrict, y_dtype, y_features, sample_weighting, dataset_name)

    return sequence, None


if __name__ == '__main__':
    import sys

    path = sys.argv[1]

    sequence, n_steps = make_generator(path, 1, y_features=0)

    print(n_steps, 'steps')
    print(sequence[0])
