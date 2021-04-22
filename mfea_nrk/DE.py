import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import numpy as np
import random
import joblib
import time
import copy
import pdb
from itertools import combinations
from datetime import datetime

from utils.input import WusnInput
from constructor.nrk import Nrk
from utils.logger import init_log
from utils import config


N_GENS = config.N_GENS
POPULATION_SIZE = config.POP_SIZE
F = config.DE_F
CR = config.DE_CR

def init_individual(num_of_relays, num_of_sensors):
    length = 2 * (num_of_sensors + num_of_relays + 1)
    individual = list(np.random.uniform(0, 1, size=(length,)))
    return individual

def run_de(inp: WusnInput, flog, logger = None, is_hop=True):    
    max_relays = 14
    max_hops = 8

    constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)
    dimension = 2 * (inp.num_of_relays + inp.num_of_sensors + 1)
    population = []
    gen_fitness = []
    for _ in range(POPULATION_SIZE):
        population.append(list(np.random.uniform(0, 1, size=(dimension,))))
        gen_fitness.append(constructor.get_loss(population[-1]))

    best_individual = copy.deepcopy(population[0])
    best_fitness = constructor.get_loss(best_individual)

    for g in range(POPULATION_SIZE):
        flog.write(f'GEN {g} time {int(time.time())}\n')
        
        for i in range(POPULATION_SIZE):
            candidates = list(range(POPULATION_SIZE))
            candidates.remove(i)
            selected_agents = np.random.choice(candidates, 3, replace=False)

            x_1 = population[selected_agents[0]]
            x_2 = population[selected_agents[1]]
            x_3 = population[selected_agents[2]]

            v_donor = [tmp1-F*(tmp2-tmp3) for tmp1, tmp2, tmp3 in zip(x_1, x_2, x_3)]
            certain_index = np.random.randint(0, dimension)
            crossover_probs = (np.random.uniform(0, 1, size=(dimension,)) < CR).astype(np.int64)
            crossover_probs[certain_index] = 1

            v_trial = [tmp2 if tmp3 else tmp1 for tmp1, tmp2, tmp3 in zip(population[i], v_donor, crossover_probs)]
            
            trial_fitness = constructor.get_loss(v_trial)
            target_fitness = gen_fitness[i]

            if trial_fitness <= target_fitness:
                population[i] = v_trial
                gen_fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_individual = v_trial[:]

        father, _ = constructor.decode_genes(best_individual)
        best_fitness = constructor.get_loss(best_individual)
        flog.write(f'{list(best_individual)}\t{father}\t{best_fitness}\n')

    if best_fitness > 10:
        return False
    return True

def solve(fn, pas=1, logger=None, is_hop=True, datadir='data/hop', logdir='results/de/hop'):
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}]solving {fn} pas {pas}')
    path = os.path.join(datadir, fn)
    os.makedirs(logdir, exist_ok=True)
    flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')

    inp = WusnInput.from_file(path)

    flog.write(f'{fn}\n')
    while not run_de(inp, flog, logger=logger, is_hop=is_hop):
        flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')
        flog.write(f'{fn}\n')

    print(f'done solved {fn}')
    

if __name__ == "__main__":
    logger = init_log()
    os.makedirs('results/de/hop', exist_ok=True)
    os.makedirs('results/de/layer', exist_ok=True)

    pases = []
    tests = []
    is_hops = []

    for i in range(5):
        rerun_hop = [tmp for tmp in os.listdir('data/small/hop') if not (('uu' in tmp and 'r50' in tmp and '_40' in tmp) or ('uu' not in tmp and ('r50' in tmp or '_40' in tmp))) and tmp != '.DS_Store']
        rerun_hop = [tmp for tmp in rerun_hop if 'ga' in tmp and 'r25' in tmp]

        tests = tests + rerun_hop
        is_hops = is_hops + ['small/hop'] * len(rerun_hop)
        pases = pases + [i] * len(rerun_hop)

        rerun_layer = [tmp for tmp in os.listdir('data/small/layer') if not (('no' in tmp or 'ga' in tmp) and 'r50' in tmp) and tmp != '.DS_Store']
        rerun_layer = [tmp for tmp in rerun_layer if 'ga' in tmp and 'r25' in tmp]

        tests = tests + rerun_layer
        is_hops = is_hops + ['small/layer'] * len(rerun_layer)
        pases = pases + [i] * len(rerun_layer)

    print(len(tests))

    joblib.Parallel(n_jobs=4)(
        joblib.delayed(solve)(fn, pas=pas, logger=logger, is_hop=True if 'hop' in is_hop else False, datadir=f'data/{is_hop}', logdir=f'results/de/{is_hop}') for \
            (pas, is_hop, fn) in zip(pases, is_hops, tests)
    )

