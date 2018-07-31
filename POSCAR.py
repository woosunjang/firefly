from __future__ import division, print_function
from collections import OrderedDict, deque
from B_C_M import getDistance
import numpy as np
import itertools


class FileWriter:
    def __init__(self, file):
        self.file = file

    def write(self, item):
        self.file.write(item + "\n")

    def close(self):
        self.file.close()


class POSCAR:
    def __init__(self, atoms, coord, lattice, name, scaling, selective, s_matrix):
        self.atoms = atoms
        self.coord = coord
        self.lattice = lattice
        self.type = 'Cartesian'
        self.name = name
        self.scaling = scaling
        self.selective = selective
        self.s_matrix = s_matrix
        self.ATOMS = OrderedDict()
        self.organize_by_atom()

        self.index_to_atom = {}
        self.atom_to_index = {}
        for idx, atm in enumerate(self.atoms.keys()):
            self.index_to_atom[idx] = atm
            self.atom_to_index[atm] = idx

    def organize_by_atom(self):
        idx, idx2 = 0, 0
        for atom, num in self.atoms.items():
            idx += idx2
            idx2 += num
            self.ATOMS[atom] = self.coord[idx:idx2]

    def print_poscar(self, filename):
        file = open(filename, 'w')
        f = FileWriter(file)
        f.write(self.name)
        f.write(self.scaling)
        for i in self.lattice:
            f.write(" ".join(map(str, i)))
        f.write(" ".join(self.atoms.keys()))
        f.write(" ".join(map(str, self.atoms.values())))
        if self.selective:
            f.write("Selective Dynamics")
        f.write(self.type)

        if self.selective:
            for i in range(len(self.coord)):
                f.write(" ".join(map(str, list(self.coord[i]) + self.s_matrix[i])))
        else:
            for i in self.coord:
                f.write(" ".join(map(str, i)))

        f.close()

    def get_max_height_of_atom(self, atom):
        # matrix
        return max([i[2] for i in self.ATOMS[atom]])

    def update_atom(self, atom):
        idx = 0
        for i in self.atoms.keys():
            if i == atom:
                break
            else:
                idx += self.atoms[i]
        for i in self.ATOMS[atom]:
            self.coord[idx] = i
            idx += 1


def post_process(line_list, selective, direct, l_parameter):
    coord = deque()
    sel = deque() if selective else None
    if selective:
        for line in line_list:
            tmp = line.split()
            tmp_coord = np.array(list(map(float, tmp[:3])))
            if len(tmp_coord) == 0:
                break
            if direct:

                coord.append(list(np.matmul(tmp_coord, l_parameter)))
            else:
                coord.append(tmp_coord)
            sel.append(tmp[3:])
    else:
        for line in line_list:
            tmp_coord = np.array(list(map(float, line.split())))
            if len(tmp_coord) == 0:
                break
            if direct:

                coord.append(list(np.matmul(tmp_coord, l_parameter)))
            else:
                coord.append(tmp_coord)
    return coord, sel


def read_poscar(file):
    line_list = file.read().splitlines()
    title = line_list[0]
    scaling = line_list[1]
    lattice = [list(map(float, i.split())) for i in line_list[2:5]]

    l_parameter = np.array(lattice)

    tmp = line_list[5].split(), list(map(int, line_list[6].split()))
    atoms = OrderedDict((tmp[0][i], tmp[1][i]) for i in range(len(tmp[0])))

    selective = False
    if line_list[7].upper()[0] == 'S':
        selective = True
    idx = 8 if selective else 7
    direct = True if line_list[idx].upper()[0] == 'D' else False

    coord, sel = post_process(line_list[idx + 1:], selective, direct, l_parameter)
    coord = [i for i in coord]
    sel_matrix = [i for i in sel] if selective else sel
    file.close()
    return POSCAR(atoms, coord, lattice, title, scaling, selective, sel_matrix)


def rearrange(atoms1, atoms2):
    distance_matrix = np.zeros((len(atoms1), len(atoms1)))
    tmp = [0 for i in range(len(atoms1))]
    for i in itertools.product(range(len(atoms1)), repeat=2):
        distance_matrix[i] = getDistance(np.array(atoms1[i[0]]), np.array(atoms2[i[1]]))

    for _ in range(len(atoms1)):
        x, y = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
        tmp[y] = x
        distance_matrix[x] = np.full(len(atoms1), np.inf)
        distance_matrix[:, y] = np.full(len(atoms1), np.inf)
    new_m = [None for i in range(len(atoms1))]
    for i, j in enumerate(tmp):
        new_m[i] = atoms1[j]
    return new_m
