from abc import abstractmethod
from copy import deepcopy
from random import uniform
import numpy as np


atom_radius = {'P': 0.98, 'Bi': 1.43}


def getdistance(x1, x2):
    return np.sqrt(sum(map(lambda x, y: (x - y) ** 2, x1, x2)))


# minimum_distance_matrix = None
def default_minimum_distance(index_to_atom):
    size = len(index_to_atom)
    minimum_distance_matrix = [[0 for i in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                minimum_distance_matrix[i][j] = 2 * atom_radius[index_to_atom[i]]
            else:
                minimum_distance_matrix[i][j] = atom_radius[index_to_atom[i]] + atom_radius[index_to_atom[j]]
    return minimum_distance_matrix


@abstractmethod
def change_minimum_distance(matrix):
    return


Bond_Ch_Matrix = []


def is_symmetry_valid(matrix):
    global Bond_Ch_Matrix
    valid = True
    if valid:
        Bond_Ch_Matrix.append(matrix)
    return valid


@abstractmethod
def get_bcm(poscar):
    return


def put_atoms(pos, atom, maxheight, fun=uniform):
    poscar_new = deepcopy(pos)
    limit = 4  # 4 A
    zrange = [maxheight, maxheight + limit]
    atom_idx = poscar_new.atom_to_index[atom]
    size = len(poscar_new.ATOMS[atom])

    xmat = np.matmul(np.array(poscar_new.lattice), np.transpose(np.array([[1, 0, 0]]))).flatten()[0]
    ymat = np.matmul(np.array(poscar_new.lattice), np.transpose(np.array([[0, 1, 0]]))).flatten()[1]
    dxdy = np.array([[xmat, 0, 0], [-xmat, 0, 0], [0, ymat, 0], [0, -ymat, 0], [xmat, ymat, 0],
                     [xmat, -ymat, 0], [-xmat, ymat, 0], [-xmat, -ymat, 0]])

    for i in range(size):
        breakcond = False
        while not breakcond:
            # TODO: 여기 inv matrix 로 validation check 해야한다.
            new_coords = list(np.matmul(np.array([fun(0,1),fun(0,1), fun(0,1)]), poscar_new.lattice))
            new_coords[2] = fun(zrange[0], zrange[1])

            for j in range(i):
                l = poscar_new.ATOMS[atom][j]

                if getdistance(l, new_coords)<= minimum_distance_matrix[atom_idx][atom_idx]:
                    break

                l = np.array(l)
                for newL in [l + i for i in dxdy]:

                    if getdistance(newL, new_coords) <= minimum_distance_matrix[atom_idx][atom_idx]:
                        break
                else:
                    continue
                break

            else:
                atom_list_without_atom = list(poscar_new.ATOMS.keys())
                atom_list_without_atom.remove(atom)
                for ATOM_SYMBOl in atom_list_without_atom:
                    breakcond = False
                    for k in poscar_new.ATOMS[ATOM_SYMBOl]:
                        if getdistance(k, new_coords) <= \
                                minimum_distance_matrix[atom_idx][poscar_new.atom_to_index[ATOM_SYMBOl]]:
                            break

                        new_k = np.array(k)
                        for newKCORD in [new_k + i for i in dxdy]:
                            if getdistance(newKCORD, new_coords) <= \
                                    minimum_distance_matrix[atom_idx][poscar_new.atom_to_index[ATOM_SYMBOl]]:
                                break
                        else:
                            continue
                        break

                    else:
                        poscar_new.ATOMS[atom][i] = new_coords
                        breakcond = True

    poscar_new.update_atom(atom)
    return poscar_new


def get_random_structure(pos, atom, maxheight):
    valid, poscar_new = False, False
    while not valid:
        poscar_new = put_atoms(pos, atom, maxheight)
        bcm = get_bcm(poscar_new)
        valid = is_symmetry_valid(bcm)
    return poscar_new
