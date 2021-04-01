import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import numpy as np
from itertools import combinations
import random
import joblib
import time
import copy
from datetime import datetime

from utils.input import WusnInput
from constructor.nrk import Nrk
from utils.logger import init_log
from utils import config


N_GENS = config.N_GENS
POPULATION_SIZE = config.POP_SIZE
W = config.PSO_W
C1 = config.PSO_C1
C2 = config.PSO_C2

class Particle:
    def __init__(self, dimension):
        self.dimension = dimension
        self.position = None
        self.best_position = None
        self.velocity = None
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')

        self.initialize()

    def initialize(self):
        self.position = np.random.uniform(0, 1, size=(self.dimension,))
        self.velocity = np.random.uniform(-1, 1, size=(self.dimension,))
        self.best_position = copy.deepcopy(self.position)

    def evaluate(self, constructor):
        self.current_fitness = constructor.get_loss(self.position)
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_position = copy.deepcopy(self.position)

    def update_velocity(self, best_g):
        r1 = np.random.uniform(0, 1, size=(self.dimension, ))
        r2 = np.random.uniform(0, 1, size=(self.dimension, ))

        cognitive_velocity = C1 * r1 * (self.best_position - self.position)
        social_velcity = C2 * r2 * (best_g - self.position)
        self.velocity = W * self.velocity + cognitive_velocity + social_velcity

    def update_position(self):
        self.position = self.position + self.velocity

def run_pso(inp: WusnInput, flog, logger = None, is_hop=True):    
    max_relays = 14
    max_hops = 8

    constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)
    dimension = 2 * (inp.num_of_relays + inp.num_of_sensors + 1)
    swarm = []

    for _ in range(POPULATION_SIZE):
        swarm.append(Particle(dimension))

    swarm[0].evaluate(constructor)
    best_particle = copy.deepcopy(swarm[0].position)
    best_fitness = swarm[0].current_fitness

    for g in range(N_GENS):
        flog.write(f'GEN {g} time {int(time.time())}\n')

        for i in range(POPULATION_SIZE):
            swarm[i].evaluate(constructor)

            if swarm[i].current_fitness < best_fitness:
                best_particle = copy.deepcopy(swarm[i].position)
                best_fitness = swarm[i].current_fitness
        
        for i in range(POPULATION_SIZE):
            swarm[i].update_velocity(best_particle)
            swarm[i].update_position()

        father, _ = constructor.decode_genes(best_particle)
        best_fitness = constructor.get_loss(best_particle)

        flog.write(f'{list(best_particle)}\t{father}\t{best_fitness}\n')

    if best_fitness > 10:
        return False
    return True

def solve(fn, pas=1, logger=None, is_hop=True, datadir='data/hop', logdir='results/pso/hop'):
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}]solving {fn} pas {pas}')
    path = os.path.join(datadir, fn)
    os.makedirs(logdir, exist_ok=True)
    flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')

    inp = WusnInput.from_file(path)

    # logger.info("prepare input data from path %s" % path)
    # logger.info("num generation: %s" % N_GENS)
    # logger.info("population size: %s" % POPULATION_SIZE)
    # logger.info("crossover probability: %s" % CXPB)
    # logger.info("mutation probability: %s" % MUTPB)

    flog.write(f'{fn}\n')
    while not run_pso(inp, flog, logger=logger, is_hop=is_hop):
        flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')
        flog.write(f'{fn}\n')

    print(f'done solved {fn}')
    

if __name__ == "__main__":
    logger = init_log()
    os.makedirs('results/pso/hop', exist_ok=True)
    os.makedirs('results/pso/layer', exist_ok=True)

    pases = []
    tests = []
    is_hops = []

    for i in range(5):
        rerun_hop = [tmp for tmp in os.listdir('data/small/hop') if not (('uu' in tmp and 'r50' in tmp and '_40' in tmp) or ('uu' not in tmp and ('r50' in tmp or '_40' in tmp))) and tmp != '.DS_Store']
        # rerun_hop = [tmp for tmp in os.listdir('data/small/hop') if 'ga' in tmp and '_0' in tmp and 'r25' in tmp]
        
        tests = tests + rerun_hop
        is_hops = is_hops + ['small/hop'] * len(rerun_hop)
        pases = pases + [i] * len(rerun_hop)

        rerun_layer = [tmp for tmp in os.listdir('data/small/layer') if not (('no' in tmp or 'ga' in tmp) and 'r50' in tmp) and tmp != '.DS_Store']
        # rerun_layer = [tmp for tmp in os.listdir('data/small/layer') if 'ga' in tmp and 'r25' in tmp]

        tests = tests + rerun_layer
        is_hops = is_hops + ['small/layer'] * len(rerun_layer)
        pases = pases + [i] * len(rerun_layer)

    print(len(tests))

    joblib.Parallel(n_jobs=4)(
        joblib.delayed(solve)(fn, pas=pas, logger=logger, is_hop=True if 'hop' in is_hop else False, datadir=f'data/{is_hop}', logdir=f'results/pso/{is_hop}') for \
            (pas, is_hop, fn) in zip(pases, is_hops, tests)
    )

