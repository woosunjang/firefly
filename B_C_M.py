
from copy import deepcopy
from random import uniform
import numpy as np


atom_radius = {'P': 0.98, 'Bi': 1.43}

def getDistance(x1, x2):
    return np.sqrt(sum(map(lambda x,y: (x-y)**2 , x1, x2)))

#
minimum_distance_matrix = None
def default_minimum_distance(index_to_atom):
    size = len(index_to_atom)
    minimum_distance_matrix = [[0 for i in range(size)]for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i==j:
                minimum_distance_matrix[i][j] = 2*atom_radius[index_to_atom[i]]
            else:
                minimum_distance_matrix[i][j] = atom_radius[index_to_atom[i]]+ atom_radius[index_to_atom[j]]
    return minimum_distance_matrix


def change_minimum_distance(matrix):
    #abstarct
    pass

# abstract
Bond_Ch_Matrix = []
def is_symmetry_valid(matrix):
    global Bond_Ch_Matrix
    valid = True
    if valid:
        Bond_Ch_Matrix.append(matrix)
    return valid
def getBondChMatrix(POSCAR):
    return 0
#

def put_Atoms(POS, atom, maxheight, fun = uniform):
    POSCAR_new = deepcopy(POS)
    limit = 4  # 4 A
    zRange = [maxheight, maxheight + limit]
    atom_idx = POSCAR_new.atom_to_index[atom]
    size = len(POSCAR_new.ATOMS[atom])

    # Change
    X = np.matmul(np.array(POSCAR_new.lattice), np.transpose(np.array([[1, 0, 0]]))).flatten()[0]
    Y = np.matmul(np.array(POSCAR_new.lattice), np.transpose(np.array([[0, 1, 0]]))).flatten()[1]
    dxdy = np.array([[X, 0, 0], [-X, 0, 0], [0, Y, 0], [0, -Y, 0], [X, Y, 0], [X, -Y, 0], [-X, Y, 0], [-X, -Y, 0]])

    for i in range(size):
        BREAK = False

        while not BREAK:
            # TODO: 여기 inv matrix 로 validation check 해야한다.
            new_coords = list(np.matmul(np.array([fun(0,1),fun(0,1), fun(0,1)]),POSCAR_new.lattice ))
            new_coords[2] = fun(zRange[0],zRange[1])

            for j in range(i):
                L = POSCAR_new.ATOMS[atom][j]

                if getDistance(L, new_coords)<= minimum_distance_matrix[atom_idx][atom_idx]:
                    break
                # Change
                L = np.array(L)
                for newL in [L+i for i in dxdy]:

                    if getDistance(newL, new_coords) <= minimum_distance_matrix[atom_idx][atom_idx]:
                        break
                else:
                    continue
                break

            else:
                atom_list_without_atom = list(POSCAR_new.ATOMS.keys())
                atom_list_without_atom.remove(atom)
                for ATOM_SYMBOl in atom_list_without_atom:
                    BREAK = False
                    for k in POSCAR_new.ATOMS[ATOM_SYMBOl]:
                        if getDistance(k, new_coords)<= minimum_distance_matrix[atom_idx][POSCAR_new.atom_to_index[ATOM_SYMBOl]]:
                            break
                        # Change
                        newK = np.array(k)
                        for newKCORD in [newK+i for i in dxdy]:
                            if getDistance(newKCORD, new_coords) <= minimum_distance_matrix[atom_idx][POSCAR_new.atom_to_index[ATOM_SYMBOl]]:
                                break
                        else:
                            continue
                        break

                    else:
                        POSCAR_new.ATOMS[atom][i] = new_coords

                        BREAK = True


    POSCAR_new.update_atom(atom)
    POSCAR_new.ATOMS[atom] = sorted(POSCAR_new.ATOMS[atom])
    return POSCAR_new

def getRandomStructure(POS,atom, maxheight):
    valid, POSCAR_new = False, False
    while not valid:
        POSCAR_new = put_Atoms(POS, atom, maxheight)
        B_C_M = getBondChMatrix(POSCAR_new)
        valid = is_symmetry_valid(B_C_M)
    return POSCAR_new


