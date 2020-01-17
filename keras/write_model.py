#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import importlib

model = sys.argv[1]

module = importlib.import_module('models.' + model)

module.write_model(sys.argv[2:])
