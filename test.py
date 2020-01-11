from constructor.nrk import Nrk
from utils.input import WusnInput
import numpy as np
import os

def init_individual(num_of_relays, num_of_sensors):
    length = 2 * (num_of_sensors + num_of_relays + 1)

    individual = list(np.random.uniform(0, 1, size=(length,)))

    return individual

max_relays = 14
max_hops = 6

def is_feasible(data_path, is_hop=True):
    inp = WusnInput.from_file(data_path)
    constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)
    
    for _ in range(1, 500):
        indiv = init_individual(inp.num_of_relays, inp.num_of_sensors)
        loss = constructor.get_loss(indiv)
        if loss < 1e8:
            return True

    return False

regen = ['ga-dem3_r25_1',
        'ga-dem8_r25_1',
        'no-dem1_r25_1',
        'no-dem2_r25_1',
        'no-dem5_r25_1',
        'no-dem8_r50_1',
        'uu-dem1_r25_1',
        'uu-dem2_r25_1',
        'uu-dem3_r25_1',
        'uu-dem3_r50_1',
        'uu-dem4_r25_1',
        'uu-dem6_r25_1',
        'ga-dem10_r25_1',
        'ga-dem2_r25_1',
        'ga-dem3_r25_1',
        'ga-dem4_r25_1',
        'no-dem5_r25_1',
        'no-dem8_r25_1',
        'uu-dem10_r50_1',
        'uu-dem1_r25_1',
        'uu-dem5_r25_1']
notok = []
regen = ['ga-dem3_r25_1']
for te in regen:
    layer_path = os.path.join('data/layer', te + '.json')
    hop_path1 = os.path.join('data/hop', te + '_0.json')
    hop_path2 = os.path.join('data/hop', te + '_40.json')
    if is_feasible(layer_path, is_hop=False) and is_feasible(hop_path1, is_hop=True) and is_feasible(hop_path2, is_hop=True):
        print('ok', te)
    else:
        print('not ok', te)
        notok.append(te)


print([tmp + '.json' for tmp in notok])