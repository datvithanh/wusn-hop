import os, sys
import copy
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import random
import numpy as np
import joblib
import time

from utils.input import WusnInput
from constructor.binary import Layer
from constructor.nrk import Nrk
from utils.logger import init_log

N_GENS = 100
POP_SIZE = 300
CXPB = 0.8
MUTPB = 0.2
TERMINATE = 30

def init_individual(num_of_relays, num_of_sensors):
    length = 2 * (num_of_sensors + num_of_relays + 1)

    individual = list(np.random.uniform(0, 1, size=(length,)))

    return individual

def run_ga(hop_inp: WusnInput, layer_inp: WusnInput, flog, logger=None):
    if logger is None:
        raise Exception("Error: logger is None!")

    # logger.info("Start!")
    num_of_relays = 14
    max_hop = 8

    hopConstructor = Nrk(hop_inp, max_relay=num_of_relays, is_hop=True, hop=max_hop)
    layerContructor = Nrk(layer_inp, max_relay=num_of_relays, is_hop=False, hop=1000)

    def factorial_rank(pop):
        factorial_cost = [(hopConstructor.get_loss(indi), layerContructor.get_loss(hopConstructor.transform_genes(indi, layer_inp.num_of_relays))) for indi in pop]

        # print('hop', min([tmp[0] for tmp in factorial_cost]))
        # print('layer', min([tmp[1] for tmp in factorial_cost]))
        # print('inf hop', sum([tmp[0] > 10 for tmp in factorial_cost]))
        # print('inf layer', sum([tmp[1] > 10 for tmp in factorial_cost]))
        
        rank_hop = zip(np.argsort([tmp[0] for tmp in factorial_cost]), range(len(pop)))
        rank_layer = zip(np.argsort([tmp[1] for tmp in factorial_cost]), range(len(pop)))

        rank_hop = [tmp[-1] for tmp in sorted(rank_hop, key=lambda x: x[0])]
        rank_layer = [tmp[-1] for tmp in sorted(rank_layer, key=lambda x: x[0])]

        factorial_rank = [rank_hop, rank_layer]
        return factorial_rank

    def skill_factor(pop):
        pop_factorial_rank = factorial_rank(pop)
        pop_skill_factor = [0 if pop_factorial_rank[0][i] <= pop_factorial_rank[1][i] else 1 for i in range(len(pop))]

        pop_scalar_fitness = [1/(min(pop_factorial_rank[0][i] + 1, pop_factorial_rank[1][i] + 1)) for i in range(len(pop))]

        # factorial rank << -> scalar fitness >> 
        return pop_skill_factor, pop_scalar_fitness

    def crossover(ind1, ind2, indpb=0.2):
        r1, r2 = np.random.randint(0, len(ind1)), np.random.randint(0, len(ind1))
        r1, r2 = min(r1, r2), max(r1, r2)
        
        ind1[:r1], ind2[:r1] = ind2[:r1], ind1[:r1]
        ind1[r2:], ind2[r2:] = ind2[r2:], ind1[r2:]

        avg = [(tmp1 + tmp2)/2 for tmp1, tmp2 in zip(ind1[r1:r2], ind2[r1:r2])]
        ind1[r1:r2], ind2[r1:r2] = avg, avg

        return ind1, ind2

    def mutate(ind, mu=0, sigma=0.2, indpb=1):
        size = len(ind)

        for i in range(size):
            if np.random.random() < indpb:
                ind[i] += random.gauss(mu, sigma)

        return ind

    def assortive_mating(pop, pop_skill_factor):
        offsprings = []
        for _ in range(POP_SIZE//2):
            idx1, idx2 = np.random.randint(0, POP_SIZE - 1), np.random.randint(0, POP_SIZE - 1)
            p1, p2 = copy.deepcopy(pop[idx1]), copy.deepcopy(pop[idx2])

            if pop_skill_factor[idx1] == pop_skill_factor[idx2] and np.random.random() < CXPB:
                offs1, offs2 = crossover(p1, p2)
                offsprings.extend([offs1, offs2])
            else:
                offs1, offs2 = mutate(p1), mutate(p2)
                offsprings.extend([offs1, offs2])

        return offsprings

    def selection(pop, pop_skill_factor, pop_scalar_fitness):
        ranking = sorted([(idx, value) for idx, value in enumerate(pop_scalar_fitness)], key=lambda x: -x[-1])
        new_pop, new_pop_skill_factor = [], []
        
        for i in range(POP_SIZE):
            new_pop.append(pop[ranking[i][0]])
            new_pop_skill_factor.append(pop_skill_factor[ranking[i][0]])

        return new_pop, new_pop_skill_factor

    pop = [init_individual(hop_inp.num_of_relays, hop_inp.num_of_sensors) for _ in range(POP_SIZE)]

    pop_skill_factor, _ = skill_factor(pop)

    offspring_pop = assortive_mating(pop, pop_skill_factor)

    hop_individual, layer_individual = None, None


    for g in range(N_GENS):
        # logger.info(f"Generation {g}")
        flog.write(f'GEN {g} time {int(time.time())}\n')

        offspring_pop = assortive_mating(pop, pop_skill_factor)
        
        immediate_pop = pop + offspring_pop
        
        pop_skill_factor, pop_scalar_fitness = skill_factor(immediate_pop)

        pop, pop_skill_factor = selection(immediate_pop, pop_skill_factor, pop_scalar_fitness)

        if pop_skill_factor[0] == 0:
            hop_individual, layer_individual = pop[0], pop[1]
        else:
            hop_individual, layer_individual = pop[1], pop[0]

        hop_obj = hopConstructor.get_loss(hop_individual)
        layer_obj = layerContructor.get_loss(hopConstructor.transform_genes(layer_individual, layer_inp.num_of_sensors))

        hop_father, hop_childcount, _ = hopConstructor.decode_genes(hop_individual)
        layer_father, layer_childcount, _ = layerContructor.decode_genes(hopConstructor.transform_genes(layer_individual, layer_inp.num_of_sensors))

        flog.write(f'{hop_individual}\t{hop_father}\t{hop_childcount}\t{hop_obj}\n{hopConstructor.transform_genes(layer_individual, layer_inp.num_of_sensors)}\t{layer_father}\t{layer_childcount}\t{layer_obj}\n')

def solve(fn, pas=1, logger=None, hop_dir='./data/hop', layer_dir='./data/layer'):
    print(f'solving {fn} pas {pas}')
    layer_fn = '_'.join(fn.split('_')[:-1]) + '.json'

    hop_path = os.path.join(hop_dir, fn)
    layer_path = os.path.join(layer_dir, layer_fn)

    # logger.info(f"prepare input data from path {hop_path} and {layer_path}")
    hop_inp = WusnInput.from_file(hop_path)
    layer_inp = WusnInput.from_file(layer_path)
    # logger.info("num generation: %s" % N_GENS)
    # logger.info("population size: %s" % POP_SIZE)
    # logger.info("crossover probability: %s" % CXPB)
    # logger.info("mutation probability: %s" % MUTPB)
    # logger.info("run GA....")

    flog = open(f'results/mfea/{fn[:-5]}_{pas}.txt', 'w+')

    flog.write(f'{fn} {layer_fn}\n')

    run_ga(hop_inp, layer_inp, flog, logger)
    print(f'done solved {fn}')
    

if __name__ == '__main__':
    logger = init_log()
    os.makedirs('results/mfea', exist_ok=True)
    hop_dir = './data/hop'
    layer_dir = './data/layer'

    for i in range(10):
        joblib.Parallel(n_jobs=8)(
            joblib.delayed(solve)(fn, pas=i, logger=logger) for fn in sorted(os.listdir(hop_dir))
            # joblib.delayed(solve)(fn, logger=logger) for fn in sorted(['ga-dem2_r25_1_0.json', 'ga-dem2_r25_1_40.json', 'ga-dem4_r25_1_0.json', 'ga-dem4_r25_1_40.json'])
        )
