def make_generator(paths, batch_size):
    while True:
        random.shuffle(paths)

        trees = [uproot.open(path)['tree'] for path in paths]

        #for data in uproot.iterate(paths, 'tree', ['x', 'n', 'y']):
        for tree in trees:
            data = tree.arrays(['x', 'n', 'y'])

            start = 0
            end = batch_size
            while True:
                yield [data['x'][start:end], data['n'][start:end]], data['y'][start:end]

                if end >= data['x'].shape[0]:
                    break

                start = end
                end += batch_size
