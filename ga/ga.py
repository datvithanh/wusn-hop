import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

from deap import base, creator, tools, algorithms
import numpy as np
from itertools import combinations
import random

from utils.input import WusnInput
from constructor.binary import Hop
from utils.logger import init_log

N_GENS = 200
POPULATION_SIZE = 300
CXPB = 0.8
MUTPB = 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.,))
FitnessMin = creator.FitnessMin
creator.create("Individual", list, fitness=FitnessMin)

def init_individual(num_of_relays, max_relays=4):
    indiviadual = [0] * num_of_relays
    relay_indices = random.sample(list(range(num_of_relays)), max_relays)
    
    for index in relay_indices:
        indiviadual[index] = 1

    return creator.Individual(indiviadual)

def get_fitness(individual, constructor):
    return constructor.get_loss(individual)

def run(inp: WusnInput, logger = None):    
    max_relays = 5
    max_hops = 6

    toolbox = base.Toolbox()

    constructor = Hop(inp, max_relays, max_hops)

    toolbox.register("individual", init_individual, inp.num_of_relays, max_relays)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=20)
    toolbox.register("evaluate", get_fitness, constructor=constructor)

    pop = toolbox.population(POPULATION_SIZE)
    best_individual = toolbox.clone(pop[0])
    logger.info("init best individual: %s, fitness: %s" % (best_individual, toolbox.evaluate(best_individual)))

    for g in range(N_GENS):
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

        logger.info("Min value this pop %d : %f " % (g, min_value))

    logger.info("Finished! Best individual: %s, fitness: %s" % (best_individual, toolbox.evaluate(best_individual)))

    return best_individual

if __name__ == "__main__":
    path = "data/hop/ga-dem5_r25_1_0.json"
    inp = WusnInput.from_file(path)

    logger = init_log()
    logger.info("prepare input data from path %s" % path)
    logger.info("num generation: %s" % N_GENS)
    logger.info("population size: %s" % POPULATION_SIZE)
    logger.info("crossover probability: %s" % CXPB)
    logger.info("mutation probability: %s" % MUTPB)

    run(inp, logger=logger)
