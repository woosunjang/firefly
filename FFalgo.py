#!/home/intern/python/Python-3.5.4/local/bin/python3

from __future__ import division , print_function
from POSCAR import POSCAR, readPOSCAR
import B_C_M as BCM
import numpy as np
import POSCAR as POS
from random import uniform
from numpy.linalg import inv
import operator
from subprocess import call
import os
from time import sleep

# FF algo must be implemented
PATH = os.getcwd()
NUMNUM =0
ori =  readPOSCAR(open('POSCAR_orig','r'))
MAXHEIGHT = 0
def do_DFT(POSCAR, INCAR):
    global NUMNUM
    ## run DFT cal in Kohn
    ## abstract
    call('mkdir TEST', shell= True)
    sleep(10)
    move = 'cp job.ff POTCAR %s TEST/.'% (INCAR)
    call(move, shell= True)
    sleep(10)
    os.chdir(os.path.join(PATH, 'TEST'))
    POSCAR.print_POSCAR('POSCAR')
    incar = 'cp %s INCAR' %(INCAR)
    call(incar, shell=True)
    sleep(10)
    call('qsub job.ff', shell= True)
    sleep(10)
    dft_val = 0
    CONTCAR =0
    while True:
        if os.path.isfile("finished"):
            call('grep "Voluntary context" OUTCAR > tmp', shell= True)
            sleep(10)
            fq = open('tmp')
            L = fq.readline()
            while len(L) == 0:
                fq.close()
                POSCAR = BCM.getRandomStructure(ori, 'Bi', MAXHEIGHT)
                POSCAR.print_POSCAR('POSCAR')
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
                L = fq.readline()
            fq.close()
            call('grep "energy  without" OUTCAR | tail -1 > test', shell=True)
            sleep(10)
            ff = open('test')
            L = ff.readline().split()
            dft_val = float(L[-1])
            CONTCAR = readPOSCAR(open("CONTCAR",'r'))
            os.chdir(PATH)
            newName = 'TEST_'+str(NUMNUM)
            call('mv -f TEST '+ newName,shell=True)
            sleep(10)
            NUMNUM+=1
            ff.close()
            break
        sleep(30)

    return CONTCAR, dft_val

def optimize(POSCAR, TYPE, INCARS, INCAR_INDEX, first_step = True):
    # type 0 uses several INCAR for optimization of single structure
    # type 1 uses several
    # TODO 인카 여러개로 확장
    if TYPE is 0:
        return do_DFT(POSCAR, INCARS[INCAR_INDEX])
    else:
        f, s = 0,0
        if first_step:
            f, s = 0, 2
        else:
            f, s = 1,3
        POSCAR,val  = do_DFT(POSCAR, INCARS[f])
        return do_DFT(POSCAR, INCARS[s])


class FFalgo:
    def __init__(self, type,OrginPOSCAR,atom, maxheight_atom, INCARS):
        self.INCARS = INCARS
        self.current_INCAR_IDX = 0
        self.optimize_type = type

        self.origin = readPOSCAR(open(OrginPOSCAR,'r'))

        BCM.minimum_distance_matrix= BCM.default_minimum_distance(self.origin.index_to_atom)

        self.atom = atom
        self.maxheight = self.origin.get_max_height_of_atom(maxheight_atom)
        self.maxheight_limit = 4 + self.maxheight# 4angstrom

        global MAXHEIGHT
        MAXHEIGHT = self.maxheight

        self.POSCARS = []
        self.coeffs = []
        self.algo_popgeneration = []
        self.rate =0

    def initialize_values(self, alpha, beta, gamma, maxpop, maxgeneration, rate):
        # recommendation alpha : 0 beta : 0.9 gamma 0.1
        self.coeffs = [alpha,beta,gamma]
        self.algo_popgeneration = [maxpop,maxgeneration]
        self.rate = rate
        self.POSCARS = [None for i in range(maxpop)]

    def makerandomStructures(self,idx):
        new_POSCAR = BCM.getRandomStructure(self.origin, self.atom, self.maxheight)
        self.POSCARS[idx] = [new_POSCAR, 0]

    def movement_coords_FF(self,atom1, atom2):
        dis = BCM.getDistance(atom1, atom2)
        value = self.coeffs[1] * np.exp(-self.coeffs[2]*dis*dis)
        L = [0,0,0]
        for i in range(3):
            L[i] = atom1[i] + value*(atom2[i]- atom1[i])
        return L

    def move_add_randomness(self,coords):
        new_coords = []
        while True:
            new_coords = [coords[0]+uniform(-self.coeffs[0],self.coeffs[0]),
                          coords[1]+ uniform(-self.coeffs[0], self.coeffs[0]), max(self.maxheight,
                        coords[2]+ uniform(-self.coeffs[0],self.coeffs[0]))]
            valid_newcoords = np.matmul(np.array(new_coords), inv(np.array(self.origin.lattice)))
            for i in valid_newcoords:
                if not(0<= i<= 1):
                    break
            else:
                break

        return new_coords

    def move_FF(self, POSCAR, POSCAR2):
        # for i in POSCAR.atoms
        size = len(POSCAR.ATOMS[self.atom])
        atom_idx = POSCAR.atom_to_index[self.atom]

        # Change
        X = np.matmul(np.array(POSCAR.lattice), np.transpose(np.array([[1, 0, 0]]))).flatten()[0]
        Y = np.matmul(np.array(POSCAR.lattice), np.transpose(np.array([[0, 1, 0]]))).flatten()[1]
        dxdy = np.array([[X, 0, 0], [-X, 0, 0], [0, Y, 0], [0, -Y, 0], [X, Y, 0], [X, -Y, 0], [-X, Y, 0], [-X, -Y, 0]])

        for i in range(size):
            BREAK = False
            new_coord_orig = self.movement_coords_FF(POSCAR.ATOMS[self.atom][i], POSCAR2.ATOMS[self.atom][i])
            UNCHANGED = True
            while not BREAK:
                if UNCHANGED:
                    new_coords = new_coord_orig
                else:
                    new_coords = self.move_add_randomness(new_coord_orig)
                for j in range(i):
                    L = POSCAR.ATOMS[self.atom][j]
                    if BCM.getDistance(L, new_coords) <= BCM.minimum_distance_matrix[atom_idx][atom_idx]:
                        UNCHANGED = False
                        break

                    # Change
                    L = np.array(L)
                    for newCORD in [L + i for i in dxdy]:
                        if BCM.getDistance(newCORD, new_coords) <= BCM.minimum_distance_matrix[atom_idx][atom_idx]:
                            UNCHANGED = False
                            break
                    else:
                        continue
                    break



                else:
                    atom_list_without_atom = list(POSCAR.ATOMS.keys())
                    atom_list_without_atom.remove(self.atom)
                    for ATOM_SYMBOl in atom_list_without_atom:
                        BREAK = False
                        for k in POSCAR.ATOMS[ATOM_SYMBOl]:
                            if BCM.getDistance(k, new_coords) <= BCM.minimum_distance_matrix[atom_idx][
                                POSCAR.atom_to_index[ATOM_SYMBOl]]:
                                UNCHANGED = False
                                break

                            # Change
                            new_k = np.array(k)
                            for newKCORD in [new_k + i for i in dxdy]:
                                if BCM.getDistance(newKCORD, new_coords) <= BCM.minimum_distance_matrix[atom_idx][
                                    POSCAR.atom_to_index[ATOM_SYMBOl]]:
                                    UNCHANGED = False
                                    break
                            else:
                                continue
                            break


                        else:
                            POSCAR.ATOMS[self.atom][i] = new_coords
                            BREAK = True
        POSCAR.update_atom(self.atom)

    def set_INCAR_index(self, num_generation):
        ## change manually
        if num_generation>25:
            self.current_INCAR_IDX = 3
        elif num_generation>15:
            self.current_INCAR_IDX = 2
        elif num_generation>5:
            self.current_INCAR_IDX = 1
        elif num_generation>=0:
            self.current_INCAR_IDX = 0

    def POSCAR_optimize(self,first_step,index =-1):
        if index == -1:
            for i in range(len(self.POSCARS)):
                self.POSCARS[i] = list(optimize(self.POSCARS[i][0], self.optimize_type, self.INCARS,
                                                self.current_INCAR_IDX, first_step))
        else:
            self.POSCARS[index] = list(optimize(self.POSCARS[index][0],self.optimize_type, self.INCARS,
                                                self.current_INCAR_IDX, first_step))

    def run(self):
        OutPutList_value =[]
        Num_POSCAR = 0
        for i in range(len(self.POSCARS)):
            self.makerandomStructures(i)
        self.POSCAR_optimize(True)
        max_idx = -3
        # Chagne this value
        max_count_limit = 10
        max_count = 0
        num_generation = 0
        pop_size = self.algo_popgeneration[0]
        first_step = True
        while num_generation<self.algo_popgeneration[1] and max_count!= max_count_limit:
            print(num_generation, 'generation running')
            self.set_INCAR_index(num_generation)
            if num_generation> 15: ## change it manually
                first_step = False
            for i in range(pop_size):
                for j in range(pop_size):
                    if self.POSCARS[i][1]>self.POSCARS[j][1]:
                        self.move_FF(self.POSCARS[i][0],self.POSCARS[j][0])
                        self.POSCAR_optimize(first_step,i)
            idx_Value_List = [(i, tmp[1])for i , tmp in enumerate(self.POSCARS)]
            idx_Value_List  = sorted(idx_Value_List, key = operator.itemgetter(1))
            for i in range(self.rate):
                name = 'POSCAR_result_'+ str(Num_POSCAR)
                Num_POSCAR+=1
                OutPutList_value.append((name, self.POSCARS[i][1]))
                self.POSCARS[i][0].print_POSCAR(name)
                self.makerandomStructures(idx_Value_List[i][0])
                self.POSCAR_optimize(first_step,i)

            tt = idx_Value_List[-1][0]
            if tt == max_idx:
                max_idx = tt
                max_count  = 0
            else:
                max_count +=1

            num_generation += 1
        else:
            for i in range(len(self.POSCARS)):
                name = 'POSCAR_result_' + str(Num_POSCAR)
                Num_POSCAR += 1
                OutPutList_value.append((name, self.POSCARS[i][1]))
                self.POSCARS[i][0].print_POSCAR(name)
            if max_count== max_count_limit:
                OutPutList_value.append(["Global optima found",' '])
        f = open('RESULT_FF','w')
        for i in OutPutList_value:
            f.write(str(i[0])+" "+str(i[1])+'\n')
        f.close()


F = FFalgo(1,'POSCAR_orig','Bi','P',['INCAR_1','INCAR_2','INCAR_3','INCAR_4'])
##initialize
## run

F.initialize_values(0.1,0.9,0.1,10,10,3)

F.run()
