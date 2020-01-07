import os, sys
import copy
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import random
import numpy as np

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

def run_ga(hop_inp: WusnInput, layer_inp: WusnInput, logger=None):
    if logger is None:
        raise Exception("Error: logger is None!")

    logger.info("Start!")
    num_of_relays = 14

    hopConstructor = Nrk(hop_inp, max_relay=num_of_relays, is_hop=True)
    layerContructor = Nrk(layer_inp, max_relay=num_of_relays, is_hop=False)

    def factorial_rank(pop):
        factorial_cost = [(hopConstructor.get_loss(indi), layerContructor.get_loss(hopConstructor.transform_genes(indi, layer_inp.num_of_relays))) for indi in pop]

        print('hop', min([tmp[0] for tmp in factorial_cost]))
        print('layer', min([tmp[1] for tmp in factorial_cost]))
        print('inf hop', sum([tmp[0] > 10 for tmp in factorial_cost]))
        print('inf layer', sum([tmp[1] > 10 for tmp in factorial_cost]))
        
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
        size = min(len(ind1), len(ind2))

        for i in range(size):
            if np.random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        return ind1, ind2

    def mutate(ind, mu=0, sigma=0.2, indpb=0.2):
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

    best_individual1, best_individual2 = None, None

    for g in range(N_GENS):
        logger.info(f"Generation {g}")

        offspring_pop = assortive_mating(pop, pop_skill_factor)
        
        immediate_pop = pop + offspring_pop
        
        pop_skill_factor, pop_scalar_fitness = skill_factor(immediate_pop)

        pop, pop_skill_factor = selection(immediate_pop, pop_skill_factor, pop_scalar_fitness)

        best_individual1, best_individual2 = pop[0], pop[1]

    best_obj1 = min(hopConstructor.get_loss(best_individual1), hopConstructor.get_loss(best_individual2))
    best_obj2 = min(layerContructor.get_loss(hopConstructor.transform_genes(best_individual1, layer_inp.num_of_sensors)), layerContructor.get_loss(hopConstructor.transform_genes(best_individual2, layer_inp.num_of_sensors)))
    return best_obj1, best_obj2

if __name__ == '__main__':
    logger = init_log()
    hop_path = './data/hop'
    layer_path = './data/layer'
    with open('nrknrk.txt', 'w+') as f:
        for fn in sorted(os.listdir(hop_path))[:10]:
            hop_fn = os.path.join(hop_path, fn)
            layer_fn = os.path.join(layer_path, fn)

            logger.info(f"prepare input data from path {hop_fn} and {layer_fn}")
            hop_inp = WusnInput.from_file(hop_fn)
            layer_inp = WusnInput.from_file(layer_fn)
            logger.info("num generation: %s" % N_GENS)
            logger.info("population size: %s" % POP_SIZE)
            logger.info("crossover probability: %s" % CXPB)
            logger.info("mutation probability: %s" % MUTPB)
            # logger.info("info input: %s" % inp.to_dict())
            logger.info("run GA....")
            best_obj1, best_obj2 = run_ga(hop_inp, layer_inp, logger)

            f.write(f'{fn} {best_obj1} {best_obj2}\n')
