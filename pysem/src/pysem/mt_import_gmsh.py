#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converts gmsh msh file to own HDF5/XMF format
"""

import sys
import h5py
import os.path as osp
import numpy as np
import argparse

# try:
#     import pysem.mt
# except ImportError:
#     _modname = osp.dirname(__file__)
#     _parent = osp.abspath(_modname)
#     sys.path.append(_parent)
#     del _parent, _modname

import pysem.mt
from pysem.mt.xdmf import create_xdmf_structure
from pysem.mt.mesh_files import Mesh


class Gmsh(object):
    def __init__(self):
        self.mesh = Mesh()
        self.sections = {
            "$Nodes" : self.read_nodes,
            "$Elements" : self.read_elements,
        }
        self.elem_type_handler = {
            # 1 :    # 2-node line. 
            2 : self.handle_tri,   # 3-node triangle. 
            3 : self.handle_quad,  # 4-node quadrangle. 
            4 : self.handle_tetra, # 4-node tetrahedron. 
            5 : self.handle_hexa,  # 8-node hexahedron. 
            # 6 : # 6-node prism. 
            # 7 : # 5-node pyramid. 
            # 8 : # 3-node second order line (2 nodes associated with the vertices and 1 with the edge). 
            # 9 : # 6-node second order triangle
            # (3 nodes associated with the vertices and 3 with the edges).
            
            # 10 : self.handle_quad9, # 9-node second order quadrangle
            # (4 nodes associated with the vertices, 4 with the edges and 1 with the face). 
            # 11 : self.handle_tetra10, # 10-node second order tetrahedron
            # (4 nodes associated with the vertices and 6 with the edges). 
            12 : self.handle_hexa27, # 27-node second order hexahedron
            # (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume). 
            # 13 : # 18-node second order prism
            # (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces). 
            # 14 : # 14-node second order pyramid
            # (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face). 
            15 : self.handle_point, # 1-node point. 
            16 : self.handle_quad8, # 8-node second order quadrangle
            # (4 nodes associated with the vertices and 4 with the edges). 
            # 17 : # 20-node second order hexahedron
            # (8 nodes associated with the vertices and 12 with the edges). 
            # 18 : # 15-node second order prism
            # (6 nodes associated with the vertices and 9 with the edges). 
            # 19 : # 13-node second order pyramid
            # (5 nodes associated with the vertices and 8 with the edges). 
            # 20 : # 9-node third order incomplete triangle
            # (3 nodes associated with the vertices, 6 with the edges) 
            # 21 : # 10-node third order triangle
            # (3 nodes associated with the vertices, 6 with the edges, 1 with the face) 
            # 22 : # 12-node fourth order incomplete triangle
            # (3 nodes associated with the vertices, 9 with the edges) 
            # 23 : # 15-node fourth order triangle
            # (3 nodes associated with the vertices, 9 with the edges, 3 with the face) 
            # 24 : # 15-node fifth order incomplete triangle
            # (3 nodes associated with the vertices, 12 with the edges) 
            # 25 : # 21-node fifth order complete triangle
            # (3 nodes associated with the vertices, 12 with the edges, 6 with the face) 
            # 26 : # 4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge) 
            # 27 : # 5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge) 
            # 28 : # 6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge) 
            # 29 : # 20-node third order tetrahedron
            # (4 nodes associated with the vertices, 12 with the edges, 4 with the faces) 
            # 30 : # 35-node fourth order tetrahedron
            # (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume) 
            # 31 : # 56-node fifth order tetrahedron
            # (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume) 
            # 92 : # 64-node third order hexahedron
            # (8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume) 
            # 93 : # 125-node fourth order hexahedron
            # (8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume) 
        }
        self.tri3_mat = []
        self.tetra4_mat = []
        self.quad4_mat = []
        self.hexa8_mat = []
        self.tag_mat = 0

    def read(self, f):
        while True:
            section = f.readline().strip()
            if not section:
                break
            if not section.startswith("$"):
                continue
            parser = self.sections.get(section, self.unknown_section)
            print("Processing section: {}".format(section))
            end_section = "$End" + section[1:]
            parser(f, end_section)

    def unknown_section(self, f, end_sect):
        while True:
            line = f.readline().strip()
            if not line:
                break
            if line == end_sect:
                return
        raise RuntimeError("Reached end of file")

    def read_nodes(self, f, end_sect):
        nnodes = int(f.readline().strip())
        self.label  = np.zeros( (nnodes,), int)
        self.reverse = {}
        while True:
            line = f.readline().strip()
            if not line:
                break
            if line == end_sect:
                return
            n,x,y,z = line.split()
            k = self.mesh.add_pt(float(x), float(y), float(z))
            self.label[k] = int(n)
            self.reverse[self.label[k]] = k
            #self.reverse[k+1] = k
            
            k = k + 1
            
        raise RuntimeError("Reached end of file")

    def read_elements(self, f, end_sect):
        ne = int(f.readline().strip())
        while True:
            line = f.readline().strip()
            if not line:
                break
            if line == end_sect:
                return
            desc = [ int(k) for k in line.split() ]
            num = desc[0]
            typ = desc[1]
            tags = desc[3:3+desc[2]]
            nodes = self.renumber(desc[3+desc[2]:])
            handle_elem = self.elem_type_handler.get(typ, self.ignore_elem)
            handle_elem(num, nodes, tags)
        raise RuntimeError("Reached end of file")

    def renumber(self, nodes):
        return tuple( self.reverse[k] for k in nodes )

    def ignore_elem(self, num, nodes, tags):
        pass

    def handle_point(self, num, nodes, tags):
        pass

    def handle_tri(self, num, nodes, tags):
        self.mesh.tri3.append(nodes)
        self.tri3_mat.append(tags[self.tag_mat]);

    def handle_quad(self, num, nodes, tags):
        self.mesh.quad4.append(nodes)
        self.quad4_mat.append(tags[self.tag_mat]);

    def handle_tetra(self, num, nodes, tags):
        print('{} {} {}'.format(num, nodes, tags))
        self.mesh.tetra4.append(nodes)
        self.tetra4_mat.append(tags[self.tag_mat]);

    def handle_hexa(self, num, nodes, tags):
        self.mesh.hexa8.append(nodes)
        self.hexa8_mat.append(tags[self.tag_mat]);

    def handle_quad8(self, num, nodes, tags):
        self.mesh.quad8.append(nodes)

    def handle_hexa27(self, num, nodes, tags):
        self.mesh.hexa27.append(nodes)
    
    def save(self, fname):
        f = h5py.File(fname, "w")
        self.mesh.write_nodes(f)
        multi = 0 # indique si on gere plusieurs types d'elements
        root = "/Sem3D"
        sem_mode = True
        ELEMTYPES = "quad4 quad8 tri3 hexa8 hexa27 tetra4".split()
        for key in ELEMTYPES:
            elems = getattr(self.mesh, key)
            if elems:
                multi += 1
        if multi>1 or self.mesh.tri3 or self.mesh.tetra4:
            root = "/Mesh"
            sem_mode = False
        for key in ELEMTYPES:
            elems = getattr(self.mesh, key)
            if not elems:
                continue
            if sem_mode or key=="hexa8":
                rootname = "Sem3D"
            else:
                rootname = "/Mesh_%s" % key
            materials = getattr(self, "%s_mat" % key)
            writer = getattr(self.mesh, "write_%s" % key)
            writer(f, rootname)
            self.mesh.write_cell_data(f, rootname, "Mat", np.array(materials, int)-1)

#        if self.mesh.quad4:
#            self.mesh.write_quad4(f, "/Sem2D")
#            self.mesh.write_cell_data(f, "/Sem2D", "Mat", np.array(self.quad4_mat,int))
#        if self.mesh.quad8:
#            self.mesh.write_quad8(f, "/Sem2D")
#            self.mesh.write_cell_data(f, "/Sem2D", "Mat", np.zeros(len(self.mesh.quad8),int))
#        if self.mesh.tri3:
#            self.mesh.write_tri3(f, "/Sem2D")
#            self.mesh.write_cell_data(f, "/Sem2D", "Mat", np.array(self.tri3_mat,int))
#        if self.mesh.hexa8:
#            self.mesh.write_hexa8(f, "/Sem3D")
#            self.mesh.write_cell_data(f, "/Sem3D", "Mat", np.zeros(len(self.mesh.hexa8),int))
#        if self.mesh.hexa27:
#            self.mesh.write_hexa27(f, "/Sem3D")
#            self.mesh.write_cell_data(f, "/Sem3D", "Mat", np.zeros(len(self.mesh.hexa27),int))
#        if self.mesh.tetra4:
#            self.mesh.write_tetra4(f, "/Sem3D")
#            self.mesh.write_cell_data(f, "/Sem3D", "Mat", np.zeros(len(self.mesh.tetra4),int))
        f.close()
        create_xdmf_structure(fname)


def main():
    parser = argparse.ArgumentParser(description="Conversion gmsh vers hdf5 (sem/mka)")
    parser.add_argument("-p", help="Use physical volume as material", dest="mat", default=0, action="store_const", const=1)
    parser.add_argument("-c", help="Use color as material", dest="mat", action="store_const", const=2)
    parser.add_argument("input", help="input file in msh format")
    parser.add_argument("output", help="output file (HDF5)")
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    opt = parser.parse_args()

    mesh = Gmsh()
    mesh.tag_mat = opt.mat
    mesh.read(open(opt.input,'r'))
    mesh.save(opt.output)

        
if __name__ == "__main__":
    main()
