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
    pop[0] = creator.Individual([-0.06459972615496325, 0.7020500614516567, 0.2999437235788304, 0.4966326670713277, 0.7222873445757572, 0.1979293991846487, 0.2587336790516607, 0.2545791402642703, -0.17017905763976643, 0.4700773253036198, 0.6661876955212027, 0.09859058825861347, 0.8760858537776218, 0.9191266758882402, 0.39271165061901925, 1.09007576587227, 0.5812376480936479, 0.15991778113472926, -0.3076882669049017, 0.6181971656752797, 0.5246666936127298, 0.17539976106995048, 1.30555387670712, 1.1573643119809092, 0.3666975853766363, 0.9228964388695287, 0.537572761118004, 1.2241100863274326, 0.45784761835622806, 1.0279760162998988, 0.13556366365907235, 1.518844521400207, 0.048509539058418934, -0.2821640419056034, 0.31875267941062835, 0.2759232454604852, 1.5006529808919638, 0.51692351030274, 0.29057291394890916, 0.9818771036568412, 0.31440454490599257, 0.7343542465314495, 0.2640583032865919, 0.8633066161980704, 0.47459464226286086, 0.554760914002748, 0.5195058823634131, 1.0320205187799352, 0.4422400944862559, 0.650665371986471, 0.7559262101621603, 0.35883302215337326, 0.9771651702772391, -0.29830722595505355, 0.10295239353582078, 0.8366353266647584, 0.5452074718142536, -0.3741414515648751, 0.8514027037482446, 0.20014677494421473, 0.32556633683803865, 0.11508809961677392, 1.3421956888641435, 0.6604075789061116, -0.00268731393045174, 0.4564550062581677, 0.804714286363923, 0.14418323876761996, 0.5458252531867002, 0.2995487670347501, 0.6564431615925848, -0.0038026141380157336, 0.8574088520414125, -0.20935919070161504, 0.9959150295136916, 0.24475447628697608, 0.895787861013827, 0.4231429163932613, 0.8364434092356325, 0.6516254472750949, 0.44131526852592196, -0.17252381785028825, 0.08938644890166142, -0.13585803625993226, 0.20462456675572072, 0.9292333958271541, 0.0337906458149253, 0.10349410107319554, 0.659466263179676, 0.18737953660383144, 0.13024116632848765, 0.3146727394144371, 0.2993504840667375, 0.2232757542151785, 0.48307425014992283, 0.49279954143018717, 0.9855173851643027, -0.5545724967679906, 0.9157932644589747, -0.15377549675886493, 1.005307519888864, 0.7243456737756726, 0.6122204898271499, 0.9901658363829178, -0.5444344562973993, 0.2761071132045114, 0.06674130336460787, 0.9330226588814818, 0.14562947369373624, 1.0314861216358748, 1.1862511010799275, 0.38381226826361137, 0.5788314195183794, 0.5047550246872593, -0.1676304404472271, 0.7193675567638231, 0.6533102694687208, 0.26902592993813845, 1.3744746621764883, 0.12581003830408943, 0.6809809482830341, -0.1445106171951102, 0.08392749855078664, 0.0067253940778740084, 0.9696019882993894, 0.613629990703651, 1.2314297005666615, 0.5660184982260104, 0.847980364607053, 0.48239178886076506, 1.03783265912374, 0.4705432729143877, 0.10179890727154803, 0.31050496039985837, 0.4010725155468736, 0.5659759478662203, 0.5805182742723732, 0.5432329347809065, 0.4031100274522247, 0.5189180232048974, 0.5582370376005341, 0.6579936092926655, 0.09623536233428942, -0.050216402149378565, 0.9087491696143906, 0.6261273866037197, 0.19763250187106085, -0.008884619189837354, 0.5203290853488183, 1.4016105266881438, -0.17824146718639206, -0.014171949408687094, 0.36080852527387264, 0.21703997865006872, -0.10084841633155539, -0.051882536390723474, 0.15844594321145725, 1.0702408583785037, 0.29888778944812, 1.2761450407525028, 0.4331877280515896, 0.8598933854258856])
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

