#!/usr/local/bin/python

from __future__ import division, print_function
from POSCAR import read_poscar, rearrange
import B_C_M as BCM
import numpy as np
from random import uniform
import operator
from subprocess import call
import os
from time import sleep

# FF algo must be implemented
working_path = os.getcwd()
numnum = 0
ori = read_poscar(open('POSCAR_orig', 'r'))
max_height = 0


def do_dft(poscar, incar):
    global numnum
    # run DFT cal in Kohn
    # abstract
    call('mkdir TEST', shell=True)
    sleep(10)
    move = 'cp job.ff POTCAR %s TEST/.' % incar
    call(move, shell=True)
    sleep(10)
    os.chdir(os.path.join(working_path, 'TEST'))
    poscar.print_poscar('POSCAR')
    incar = 'cp %s INCAR' % incar
    call(incar, shell=True)
    sleep(10)
    call('qsub job.ff', shell=True)
    sleep(10)
    # dft_val = 0
    # contcar = 0
    while True:
        if os.path.isfile("finished"):
            call('grep "Voluntary context" OUTCAR > tmp', shell=True)
            sleep(10)
            fq = open('tmp')
            lines = fq.readline()
            while len(lines) == 0:
                fq.close()
                poscar = BCM.get_random_structure(ori, 'Bi', max_height)
                poscar.print_poscar('POSCAR')
                call('rm finished', shell=True)
                sleep(10)
                call('qsub job.ff', shell=True)
                sleep(10)
                while True:
                    if os.path.isfile("finished"):
                        call('grep "Voluntary context" OUTCAR > tmp', shell=True)
                        break
                    sleep(15)
                fq = open('tmp')
                lines = fq.readline()
            fq.close()
            call('grep "energy  without" OUTCAR | tail -1 > test', shell=True)
            sleep(10)
            ff = open('test')
            lines = ff.readline().split()
            dft_val = float(lines[-1])
            contcar = read_poscar(open("contcar", 'r'))
            os.chdir(working_path)
            new_name = 'TEST_' + str(numnum)
            call('mv -f TEST ' + new_name, shell=True)
            sleep(10)
            numnum += 1
            ff.close()
            break
        sleep(30)

    return contcar, dft_val


def optimize(poscar, opt_type, incars, incar_index, first_step=True):
    # type 0 uses several INCAR for optimization of single structure
    # type 1 uses several
    # TODO 인카 여러개로 확장
    if opt_type is 0:
        return do_dft(poscar, incars[incar_index])
    else:
        f, s = 0, 0
        if first_step:
            f, s = 0, 2
        else:
            f, s = 1, 3
        poscar, val = do_dft(poscar, incars[f])
        return do_dft(poscar, incars[s])


class FFalgo:
    def __init__(self, opt_type, poscar_init, atom, maxheight_atom, incars):
        self.INCARS = incars
        self.current_INCAR_IDX = 0
        self.optimize_type = opt_type

        self.origin = read_poscar(open(poscar_init, 'r'))

        BCM.minimum_distance_matrix = BCM.default_minimum_distance(self.origin.index_to_atom)

        self.atom = atom
        self.maxheight = self.origin.get_max_height_of_atom(maxheight_atom)
        self.maxheight_limit = 4 + self.maxheight  # 4angstrom

        global max_height
        max_height = self.maxheight

        self.poscars = []
        self.coeffs = []
        self.algo_popgeneration = []
        self.rate = 0

    def initialize_values(self, alpha, beta, gamma, maxpop, maxgeneration, rate):
        # recommendation alpha : 0 beta : 0.9 gamma 0.1
        self.coeffs = [alpha, beta, gamma]
        self.algo_popgeneration = [maxpop, maxgeneration]
        self.rate = rate
        self.poscars = [None for i in range(maxpop)]

    def make_random_structure(self, idx):
        new_poscar = BCM.get_random_structure(self.origin, self.atom, self.maxheight)
        self.poscars[idx] = [new_poscar, 0]

    def movement_coords_firefly(self, atom1, atom2):
        dis = BCM.get_distance(atom1, atom2)
        value = self.coeffs[1] * np.exp(-self.coeffs[2] * dis * dis)
        coord_matrix = [0, 0, 0]
        for i in range(3):
            coord_matrix[i] = atom1[i] + value * (atom2[i] - atom1[i])
        return coord_matrix

    def move_add_randomness(self, coords):
        new_coords = [coords[0] + uniform(-self.coeffs[0], self.coeffs[0]),
                      coords[1] + uniform(-self.coeffs[0], self.coeffs[0]),
                      coords[2] + uniform(-self.coeffs[0], self.coeffs[0])]

        return new_coords

    def move_firefly(self, poscar, poscar2):
        poscar.atoms[self.atom].sort(key=lambda x: x[2])
        poscar.update_atom(self.atom)
        poscar2.atoms[self.atom].sort(key=lambda x: x[2])
        poscar2.update_atom(self.atom)
        poscar.atoms[self.atom] = rearrange(poscar.atoms[self.atom], poscar2.atoms[self.atom])
        poscar.update_atom(self.atom)
        # for i in POSCAR.atoms
        size = len(poscar.atoms[self.atom])
        atom_idx = poscar.atom_to_index[self.atom]

        # Change
        x_val = np.matmul(np.array(poscar.lattice), np.transpose(np.array([[1, 0, 0]]))).flatten()[0]
        y_val = np.matmul(np.array(poscar.lattice), np.transpose(np.array([[0, 1, 0]]))).flatten()[1]
        dxdy = np.array([[x_val, 0, 0], [-x_val, 0, 0], [0, y_val, 0], [0, -y_val, 0],
                         [x_val, y_val, 0], [x_val, -y_val, 0], [-x_val, y_val, 0], [-x_val, -y_val, 0]])

        for i in range(size):
            break_cond = False
            new_coord_orig = self.movement_coords_firefly(poscar.atoms[self.atom][i], poscar2.atoms[self.atom][i])
            unchanged = True
            loop_count = 1
            while not break_cond:
                loop_count += 1
                if loop_count > 150:
                    poscar.update_atom(self.atom)
                    return

                if unchanged:
                    new_coords = new_coord_orig
                else:
                    new_coords = self.move_add_randomness(new_coord_orig)
                for j in range(i):
                    coord_matrix = poscar.atoms[self.atom][j]
                    if BCM.get_distance(coord_matrix, new_coords) <= BCM.minimum_distance_matrix[atom_idx][atom_idx]:
                        unchanged = False
                        break

                    # Change
                    coord_matrix = np.array(coord_matrix)
                    for newCORD in [coord_matrix + i for i in dxdy]:
                        if BCM.get_distance(newCORD, new_coords) <= BCM.minimum_distance_matrix[atom_idx][atom_idx]:
                            unchanged = False
                            break
                    else:
                        continue
                    break

                else:
                    atom_list_without_atom = list(poscar.atoms.keys())
                    atom_list_without_atom.remove(self.atom)
                    for ATOM_SYMBOl in atom_list_without_atom:
                        break_cond = False
                        for k in poscar.atoms[ATOM_SYMBOl]:
                            if BCM.get_distance(k, new_coords) <= \
                                    BCM.minimum_distance_matrix[atom_idx][poscar.atom_to_index[ATOM_SYMBOl]]:
                                unchanged = False
                                break

                            # Change
                            new_k = np.array(k)
                            for newkcoord in [new_k + i for i in dxdy]:
                                if BCM.get_distance(newkcoord, new_coords) <= \
                                        BCM.minimum_distance_matrix[atom_idx][poscar.atom_to_index[ATOM_SYMBOl]]:
                                    unchanged = False
                                    break
                            else:
                                continue
                            break

                        else:
                            poscar.atoms[self.atom][i] = new_coords
                            poscar.update_atom(self.atom)
                            break_cond = True
        poscar.update_atom(self.atom)

    def set_incar_index(self, num_generation):
        # change manually
        if num_generation > 25:
            self.current_INCAR_IDX = 3
        elif num_generation > 15:
            self.current_INCAR_IDX = 2
        elif num_generation > 5:
            self.current_INCAR_IDX = 1
        elif num_generation >= 0:
            self.current_INCAR_IDX = 0

    def poscar_optimize(self, first_step, index=-1):
        if index == -1:
            for i in range(len(self.poscars)):
                self.poscars[i] = list(optimize(self.poscars[i][0], self.optimize_type, self.INCARS,
                                                self.current_INCAR_IDX, first_step))
        else:
            self.poscars[index] = list(optimize(self.poscars[index][0], self.optimize_type, self.INCARS,
                                                self.current_INCAR_IDX, first_step))

    def run(self):
        outputlist_value = []
        num_poscar = 0
        for i in range(len(self.poscars)):
            self.make_random_structure(i)
        self.poscar_optimize(True)
        max_idx = -3
        # Chagne this value
        max_count_limit = 10
        max_count = 0
        num_generation = 0
        pop_size = self.algo_popgeneration[0]
        first_step = True
        while num_generation < self.algo_popgeneration[1] and max_count != max_count_limit:
            print(num_generation, 'generation running')
            call('mkdir Generation_{}'.format(num_generation), shell=True)
            os.chdir(os.path.join(working_path, 'Generation_{}'.format(num_generation)))
            for pos in self.poscars:
                pos[0].print_poscar('POSCAR_{}_{}.vasp'.format(num_generation, pos[1]))
            os.chdir(working_path)

            self.set_incar_index(num_generation)
            if num_generation > 15:  # change it manually
                first_step = False
            for i in range(pop_size):
                for j in range(pop_size):
                    if self.poscars[i][1] > self.poscars[j][1]:
                        self.move_firefly(self.poscars[i][0], self.poscars[j][0])
                        self.poscar_optimize(first_step, i)
            idx_value_list = [(i, tmp[1]) for i, tmp in enumerate(self.poscars)]
            idx_value_list = sorted(idx_value_list, key=operator.itemgetter(1))
            for i in range(self.rate):
                name = 'POSCAR_result_' + str(num_poscar)
                num_poscar += 1
                outputlist_value.append((name, self.poscars[i][1]))
                self.poscars[i][0].print_poscar(name)
                self.make_random_structure(idx_value_list[i][0])
                self.poscar_optimize(first_step, i)

            tt = idx_value_list[-1][0]
            if tt == max_idx:
                max_idx = tt
                max_count = 0
            else:
                max_count += 1

            num_generation += 1
        else:
            for i in range(len(self.poscars)):
                name = 'POSCAR_result_' + str(num_poscar)
                num_poscar += 1
                outputlist_value.append((name, self.poscars[i][1]))
                self.poscars[i][0].print_poscar(name)
            if max_count == max_count_limit:
                outputlist_value.append(["Global optima found", ' '])
        f = open('RESULT_FF', 'w')
        for i in outputlist_value:
            f.write(str(i[0]) + " " + str(i[1]) + '\n')
        f.close()


F = FFalgo(1, 'POSCAR_orig', 'Bi', 'P', ['INCAR_1', 'INCAR_2', 'INCAR_3', 'INCAR_4'])
# initialize
# run

F.initialize_values(0.4, 0.9, 0.1, 10, 10, 3)

F.run()
