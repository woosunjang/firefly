from copy import deepcopy
from random import uniform
import numpy as np
import itertools
import scipy.optimize

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

   POSCAR_new.ATOMS[atom].sort(key = lambda x: x[2])
   POSCAR_new.update_atom(atom)
   return POSCAR_new

def getRandomStructure(POS,atom, maxheight):
    valid, POSCAR_new = False, False
    while not valid:
        POSCAR_new = put_Atoms(POS, atom, maxheight)
        B_C_M = getBondChMatrix(POSCAR_new)
        valid = is_symmetry_valid(B_C_M)
    return POSCAR_new


# 0: eps, 1: sigma
eps_sigma_dict = {'Bi': {'P' : [1,1], 'Bi': [1,1]}, 'P': {'Bi': [1,1],'P':[1,1]}}


def set_eps_sigma():
    pass


def get_eps_sigma(atom1, atom2):
    return eps_sigma_dict[atom1][atom2]


def lj_potential_atoms(atom1, atom2, dis):
    eps, sigma = get_eps_sigma(atom1, atom2)
    return 4*eps*((sigma/dis)**12 - (sigma/dis)**6)

def lj_force_magnitude_atoms(atom1, atom2 , dis):
    eps, sigma = get_eps_sigma(atom1, atom2)
    if dis< 1E-08 :
        dis = 1E-08
    return 48*eps*((sigma/dis)**12 - 0.5*((sigma/dis)**6))/(dis**2)



def lj_potential_poscar(poscar):
    total_atom = poscar.atoms.keys()
    total_energy = 0
    for atom1,atom2 in itertools.combinations_with_replacement(total_atom, 2):
        if atom1 == atom2:
            for pos1, pos2 in itertools.combinations(poscar.ATOMS[atom1], 2):
                total_energy += lj_potential_atoms(atom1, atom2, getDistance(pos1, pos2))
        else:
            for pos1, pos2 in itertools.product(poscar.ATOMS[atom1], poscar.ATOMS[atom2]):
                total_energy += lj_potential_atoms(atom1, atom2, getDistance(pos1, pos2))
    return total_energy


def lj_force_for_optimize(pos, poscar, target_atom):
    pos = list(np.array(pos).reshape((-1, 3)))
    poscar.ATOMS[target_atom] = pos
    f = np.zeros((len(pos), 3))
    total_atoms = poscar.atoms.keys()
    for atom2 in total_atoms:
        if target_atom == atom2:
            for idx in range(len(pos)-1):
                for idx2 in range(idx+1, len(pos)):
                    pos_idx, pos_idx2 = np.array(pos[idx]), np.array(pos[idx2])
                    vec = pos_idx2 - pos_idx
                    mag = lj_force_magnitude_atoms(target_atom, atom2,getDistance(pos_idx, pos_idx2))
                    f[idx] -= mag*vec
                    f[idx2] += mag*vec
        else:
            for idx in range(len(pos)):
                for pos_other in poscar.ATOMS[atom2]:
                    pos_idx, pos_other = np.array(pos[idx]), np.array(pos_other)
                    vec = pos_other - pos_idx
                    mag = lj_force_magnitude_atoms(target_atom, atom2, getDistance(pos_idx, pos_other))
                    f[idx] -= mag*vec
    return f.flatten()


def lj_potential_for_optimize(pos, poscar, target_atom):
    pos = list(np.array(pos).reshape((-1,3)))
    poscar.ATOMS[target_atom] = pos
    return lj_potential_poscar(poscar)

def lj_gradient(pos, poscar, target_atom):
    return -lj_force_for_optimize(pos, poscar, target_atom)


def lj_optimize(poscar, target_atom, method='BFGS'):
    if method in ['Nelder-Mead', 'Powell']:
        jac = None
        options = {'maxiter': 200, 'disp': False}
    else:
        jac = lj_gradient
        options = {'gtol': 1E-5, 'disp': False}
    first_x = np.array(poscar.ATOMS[target_atom]).flatten()
    final_x = scipy.optimize.minimize(lj_potential_for_optimize, first_x, args=(poscar, target_atom),
                                      method=method, jac=jac, options=options)

    poscar.ATOMS[target_atom] = list(np.array(final_x.x).reshape((-1,3)))
    poscar.update_atom(target_atom)
    return poscar