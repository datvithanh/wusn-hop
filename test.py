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
    print(constructor._edges[0])
    exit(0)
    
    for _ in range(1, 2):
        indiv = init_individual(inp.num_of_relays, inp.num_of_sensors)
        loss = constructor.get_loss(indiv)
        # print(loss)
        if loss < 1e8:
            return True

    return False

for dn in ['small', 'medium', 'large']:
    for test in ['layer', 'hop']:
        for fn in os.listdir(os.path.join('data', dn, test)):
            test_path = os.path.join('data', dn, test, fn)
            is_feasible(test_path, is_hop=test=='hop')
            print(test_path, is_feasible(test_path, is_hop=test=='hop'))
# for i in range()
#     print(is_feasible('data/layer/uu-dem4_r25_1.json', is_hop=False))