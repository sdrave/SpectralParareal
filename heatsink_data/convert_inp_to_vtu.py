#!/usr/bin/env python3

import sys

import skfem as fem

filename = sys.argv[1]
assert filename.endswith('.inp')
mesh = fem.Mesh.load(filename)
mesh.save(filename.removesuffix('.inp') + '.vtu')
