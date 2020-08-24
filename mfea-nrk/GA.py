import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

from deap import base, creator, tools, algorithms
import numpy as np
from itertools import combinations
import random
import joblib
import time

from utils.input import WusnInput
from constructor.nrk import Nrk
from utils.logger import init_log

N_GENS = 100
POPULATION_SIZE = 300
CXPB = 0.8
MUTPB = 0.2

# creator.create("FitnessMin", base.Fitness, weights=(-1.,))
# FitnessMin = creator.FitnessMin
# creator.create("Individual", list, fitness=FitnessMin)

def init_individual(num_of_relays, num_of_sensors):
    length = 2 * (num_of_sensors + num_of_relays + 1)

    individual = list(np.random.uniform(0, 1, size=(length,)))

    return creator.Individual(individual)

def get_fitness(individual, constructor):
    return constructor.get_loss(individual)

def crossover(ind1, ind2, indpb=0.2):
    # size = min(len(ind1), len(ind2))
    # for i in range(size):
    #     if np.random.random() < indpb:
    #         ind1[i], ind2[i] = ind2[i], ind1[i]

    r1, r2 = np.random.randint(0, len(ind1)), np.random.randint(0, len(ind1))
    r1, r2 = min(r1, r2), max(r1, r2)
    
    ind1[:r1], ind2[:r1] = ind2[:r1], ind1[:r1]
    ind1[r2:], ind2[r2:] = ind2[r2:], ind1[r2:]

    avg = [(tmp1 + tmp2)/2 for tmp1, tmp2 in zip(ind1[r1:r2], ind2[r1:r2])]
    ind1[r1:r2], ind2[r1:r2] = avg, avg

    return ind1, ind2

def mutate(ind, mu=0, sigma=0.2, indpb=0.4):
    size = len(ind)

    for i in range(size):
        if np.random.random() < indpb:
            ind[i] += random.gauss(mu, sigma)

    return ind,

def run(inp: WusnInput, flog, logger = None, is_hop=True):    

    creator.create("FitnessMin", base.Fitness, weights=(-1.,))
    FitnessMin = creator.FitnessMin
    creator.create("Individual", list, fitness=FitnessMin)

    max_relays = 30
    max_hops = 12

    toolbox = base.Toolbox()

    constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)

    toolbox.register("individual", init_individual, inp.num_of_relays, inp.num_of_sensors)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=20)
    toolbox.register("evaluate", get_fitness, constructor=constructor)

    pop = toolbox.population(POPULATION_SIZE)
    best_individual = toolbox.clone(pop[0])
    # logger.info("init best individual: %s, fitness: %s" % (best_individual, toolbox.evaluate(best_individual)))
    
    for _ in range(100):
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        if min(fitnesses) < 1e8:
            best_individual = toolbox.clone(pop[np.argmin(fitnesses)])
            break
        
    for g in range(N_GENS):
        flog.write(f'GEN {g} time {int(time.time())}\n')
        # Selection
        offsprings = map(toolbox.clone, toolbox.select(pop, len(pop) - 1))
        # Produce offsprings
        offsprings = algorithms.varAnd(offsprings, toolbox, CXPB, MUTPB)

        # Create intermediate population
        min_value = float('inf')
        intermediate = []
        tmp = [ind for ind in offsprings]
        tmp.append(best_individual)
        fitnesses = toolbox.map(toolbox.evaluate, tmp)

        cnt = 0
        
        for ind, fit in zip(tmp, fitnesses):
            if fit == float('inf'):
                cnt += 1
                intermediate.append(best_individual)
            else:
                intermediate.append(ind)

        fitnesses = toolbox.map(toolbox.evaluate, intermediate)

        for ind, fit in zip(intermediate, fitnesses):
            ind.fitness.values = [fit]
            if min_value > fit:
                min_value = fit
                best_individual = toolbox.clone(ind)

        # b = round(min_value, 6)
        pop[:] = intermediate[:]

        father, num_child, _ = constructor.decode_genes(best_individual)
        obj = constructor.get_loss(best_individual)

        flog.write(f'{best_individual}\t{father}\t{num_child}\t{obj}\n')

        # logger.info("Min value this pop %d : %f " % (g, min_value))
    obj = constructor.get_loss(best_individual)
    if obj > 10:
        return False
    return True
    # logger.info("Finished! Best individual: %s, fitness: %s" % (best_individual, toolbox.evaluate(best_individual)))
    # return best_individual

def solve(fn, pas=1, logger=None, is_hop=True, datadir='data/hop', logdir='results/hop'):
    print(f'solving {fn} pas {pas}')
    path = os.path.join(datadir, fn)
    flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')

    inp = WusnInput.from_file(path)

    # logger.info("prepare input data from path %s" % path)
    # logger.info("num generation: %s" % N_GENS)
    # logger.info("population size: %s" % POPULATION_SIZE)
    # logger.info("crossover probability: %s" % CXPB)
    # logger.info("mutation probability: %s" % MUTPB)

    flog.write(f'{fn}\n')
    while not run(inp, flog, logger=logger, is_hop=is_hop):
        flog = open(f'{logdir}/{fn[:-5]}_{pas}.txt', 'w+')
        flog.write(f'{fn}\n')
    print(f'done solved {fn}')
    

if __name__ == "__main__":
    logger = init_log()
    os.makedirs('results/hop', exist_ok=True)
    os.makedirs('results/layer', exist_ok=True)

    # for i in range(10):
    #     joblib.Parallel(n_jobs=8)(
    #         joblib.delayed(solve)(fn, pas=i, logger=logger, is_hop=True, datadir='data/hop', logdir='results/hop') for fn in os.listdir('data/hop')
    #     )

    #     joblib.Parallel(n_jobs=8)(
    #         joblib.delayed(solve)(fn, pas=i, logger=logger, is_hop=False, datadir='data/layer', logdir='results/layer') for fn in os.listdir('data/layer')
    #     )

    rerun = list(set([tmp.replace('\n', '') for tmp in open('run_hop_total.txt', 'r').readlines()])) + \
            list(set([tmp.replace('\n', '') for tmp in open('run_layer_total.txt', 'r').readlines()]))

    pases = []
    tests = []
    is_hops = []

    for i in range(10):
        rerun_hop = [tmp for tmp in os.listdir('data/small/hop') if f'{tmp[:-5]}_{i}.txt' in rerun]

        tests = tests + rerun_hop
        is_hops = is_hops + ['small/hop'] * len(rerun_hop)
        pases = pases + [i] * len(rerun_hop)

        rerun_layer = [tmp for tmp in os.listdir('data/small/layer') if f'{tmp[:-5]}_{i}.txt' in rerun]

        tests = tests + rerun_layer
        is_hops = is_hops + ['small/layer'] * len(rerun_layer)
        pases = pases + [i] * len(rerun_layer)

    joblib.Parallel(n_jobs=16)(
        joblib.delayed(solve)(fn, pas=pas, logger=logger, is_hop=True if is_hop == 'hop' else False, datadir=f'data/{is_hop}', logdir=f'results/{is_hop}') for \
            (pas, is_hop, fn) in zip(pases, is_hops, tests)
    )

