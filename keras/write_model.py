#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import importlib

modelmod = importlib.import_module('models.%s' % sys.argv[1])

model = modelmod.make_model()

model.summary()

with open(sys.argv[2], 'w') as json_file:
    json_file.write(model.to_json())
