from copy import deepcopy
from random import uniform
import numpy as np

atom_radius = {'P': 0.98, 'Bi': 1.43}
minimum_distance_matrix = None


def get_distance(x1, x2):
    return np.sqrt(sum(map(lambda x, y: (x - y) ** 2, x1, x2)))


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


def change_minimum_distance(matrix):
    # abstarct
    pass


bcm = []


def is_symmetry_valid(matrix):
    global bcm
    valid = True
    if valid:
        bcm.append(matrix)
    return valid


def get_bcm(poscar):
    return 0


def put_atoms(pos, atom, maxheight, fun=uniform):
    poscar_new = deepcopy(pos)
    limit = 4  # 4 A
    z_range = [maxheight, maxheight + limit]
    atom_idx = poscar_new.atom_to_index[atom]
    size = len(poscar_new.atoms[atom])

    # Change
    xval = np.matmul(np.array(poscar_new.lattice), np.transpose(np.array([[1, 0, 0]]))).flatten()[0]
    yval = np.matmul(np.array(poscar_new.lattice), np.transpose(np.array([[0, 1, 0]]))).flatten()[1]
    dxdy = np.array([[xval, 0, 0], [-xval, 0, 0], [0, yval, 0], [0, -yval, 0], 
                     [xval, yval, 0], [xval, -yval, 0], [-xval, yval, 0], [-xval, -yval, 0]])

    for i in range(size):
        break_cond = False

        while not break_cond:
            # TODO: 여기 inv matrix 로 validation check 해야한다.
            new_coords = list(np.matmul(np.array([fun(0, 1), fun(0, 1), fun(0, 1)]), poscar_new.lattice))
            new_coords[2] = fun(z_range[0], z_range[1])

            for j in range(i):
                L = poscar_new.atoms[atom][j]

                if get_distance(L, new_coords) <= minimum_distance_matrix[atom_idx][atom_idx]:
                    break
                # Change
                L = np.array(L)
                for newL in [L + i for i in dxdy]:

                    if get_distance(newL, new_coords) <= minimum_distance_matrix[atom_idx][atom_idx]:
                        break
                else:
                    continue
                break

            else:
                atom_list_without_atom = list(poscar_new.atoms.keys())
                atom_list_without_atom.remove(atom)
                for ATOM_SYMBOl in atom_list_without_atom:
                    break_cond = False
                    for k in poscar_new.atoms[ATOM_SYMBOl]:
                        if get_distance(k, new_coords) <= \
                                minimum_distance_matrix[atom_idx][poscar_new.atom_to_index[ATOM_SYMBOl]]:
                            break
                        # Change
                        new_k = np.array(k)
                        for newKCORD in [new_k + i for i in dxdy]:
                            if get_distance(newKCORD, new_coords) <= \
                                    minimum_distance_matrix[atom_idx][poscar_new.atom_to_index[ATOM_SYMBOl]]:
                                break
                        else:
                            continue
                        break

                    else:
                        poscar_new.atoms[atom][i] = new_coords

                        break_cond = True

    poscar_new.atoms[atom].sort(key=lambda x: x[2])
    poscar_new.update_atom(atom)
    return poscar_new


def get_random_structure(pos, atom, maxheight):
    valid, poscar_new = False, False
    while not valid:
        poscar_new = put_atoms(pos, atom, maxheight)
        bcm_new = get_bcm(poscar_new)
        valid = is_symmetry_valid(bcm_new)
    return poscar_new
