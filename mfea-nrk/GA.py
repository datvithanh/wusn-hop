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

    max_relays = 14
    max_hops = 8

    toolbox = base.Toolbox()

    constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)

    toolbox.register("individual", init_individual, inp.num_of_relays, inp.num_of_sensors)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=20)
    toolbox.register("evaluate", get_fitness, constructor=constructor)

    pop = toolbox.population(POPULATION_SIZE)
    pop[0] = creator.Individual([0.4199851092624649, -0.15090554971361742, 0.8070322343060116, 0.538662247439568, 0.8467713291771809, 0.5816648055640273, 0.4621189408284302, -0.12686173228612524, 0.7262236366954112, 0.30343418389827104, 0.5474100700086406, 0.5607468937208169, 0.6638998089408071, 0.3061806756492076, 0.14574380590638522, 0.3277832094784092, 0.3319247866606555, 0.3195694138980062, 0.8196723185156379, 0.27173715794437175, -0.07125885301154222, 0.48031519604355255, 1.1195135969502412, 0.5404453845236301, 0.47457860101461535, -0.078448656922734, 0.830425952776821, 0.08825191026900156, 0.7624884479547053, 0.3995903263363942, 0.22847954890187874, -0.04411768824586704, 0.5497669147071365, 0.9988899707442598, 0.4400953582129487, 0.3589984370875418, 0.6725817406938885, 0.5974138635071323, 0.4900068959084322, 0.26691956587219284, 0.2728554200100852, 0.5979675991909337, 0.8294468537906475, 0.5345554504561723, 0.15444259339280614, 1.0480442061211592, 0.1417260595280501, 0.9085408316551826, 0.5795560583761996, -0.02093543537470407, 0.6084989608091936, 0.6602945252160093, 0.41340767076280893, 0.37739618906907046, 0.2929381821962787, 0.11102625906250012, 0.6048498995306106, 0.9798211661748177, 0.8259802109421867, 0.8496025658172088, 0.011711270402778078, 0.30806214367172435, 0.7277084152325154, 0.9631838550916733, 0.5643097258955456, 0.5595972204508198, 0.9153377127673942, 1.1485834112296152, 0.5002647755314499, 0.9513889493533566, 0.4083131732809723, 0.9919650486862323, 0.3288231909846656, 0.06875053230876471, 0.9855196213985017, 0.7900597852541342, 0.8950113027540498, 0.2660710214524843, 0.26583095552643926, 0.34831913363872025, 0.4627366685876016, 1.2857006561053872, 0.37475739190025964, 0.9998235650649961, -0.05112289571419926, 0.9247167958427315, 0.4948235979089267, 0.6365963809861999, 0.1226646841838541, 0.22688780074650539, 0.19068646514993542, 0.2465260037267198, 0.12224676021711223, 0.6838835484484486, 1.007831458982552, 0.0828562998497865, 0.3506301251262094, 0.23107425359547795, 0.9393602781666232, 1.0443943882877076, 0.33118408340259753, 0.08538288428633481, 0.6843347195997072, 0.4347363408286439, 0.16611503191798196, 0.5840785427983939, 0.47441205805324116, 0.127632750570716, 0.6258727696355245, 0.016250146028690793, 0.2737860689895349, 0.6005802684314449, 0.21626410496483495, 0.18145253885136764, 0.87774236860004, 0.5543114698858704, 0.6904958860483035, 0.32825918617350147, 0.1640333247542767, 0.3113549569170684, 0.18829288513579, 0.46568132664691997, 0.17071035071122614, 0.4326416994214617, 0.08153234434632775, 0.5390309512683338, 0.4301637384228997, 0.16753826513563702, 0.7203700513407917, 0.39530701670825624, 0.841830974093187, 0.9851688968516202, -0.07973753411780332, 0.35861451551089046, 0.4498133844270193, 0.17570615548718843, 0.45045876124423967, 0.6108311938643323, 0.2875404499357116, 0.40592181686946877, 0.7211833866440711, 1.0029879663117856, 1.2092952844178344, -0.04379425234734213, 0.7109828233768605, 0.8212610318076242, 0.7684945307483465, 0.8373636306326426, 1.0790615909779264, 0.798986936469271, -0.07305716682748356, 0.6390416924759551, 0.6689095065000908, -0.09730840247233677, 0.4796553969957299, -0.06492966004455575, 0.32241280094908936, 0.4209289022849708, 1.2772620367444925, 0.5859159091987678, 0.6570767879835822, 0.16648040844389977])
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

    run(inp, flog, logger=logger, is_hop=is_hop)
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

    rerun = set([tmp.replace('\n', '') for tmp in open('rerun.txt', 'r').readlines()])
    pases = []
    tests = []
    is_hops = []

    for i in range(10):
        rerun_hop = [tmp for tmp in os.listdir('data/hop') if f'{tmp[:-5]}_{i}.txt' in rerun]

        tests = tests + rerun_hop
        is_hops = is_hops + ['hop'] * len(rerun_hop)
        pases = pases + [i] * len(rerun_hop)

        rerun_layer = [tmp for tmp in os.listdir('data/layer') if f'{tmp[:-5]}_{i}.txt' in rerun]

        tests = tests + rerun_layer
        is_hops = is_hops + ['layer'] * len(rerun_layer)
        pases = pases + [i] * len(rerun_layer)

    joblib.Parallel(n_jobs=8)(
        joblib.delayed(solve)(fn, pas=pas, logger=logger, is_hop=True if is_hop == 'hop' else False, datadir=f'data/{is_hop}', logdir=f'results/{is_hop}') for \
            (pas, is_hop, fn) in zip(pases, is_hops, tests)
    )

