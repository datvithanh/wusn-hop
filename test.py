from constructor.nrk import Nrk
from utils.input import WusnInput
import numpy as np

max_relays = 14
max_hops = 6
inp = WusnInput.from_file('data/hop/uu-dem2_r25_1_0.json')
is_hop = True
inp = WusnInput.from_file('data/layer/uu-dem9_r50_1.json')
is_hop = False
constructor = Nrk(inp, max_relays, max_hops, is_hop=is_hop)

def init_individual(num_of_relays, num_of_sensors):
    length = 2 * (num_of_sensors + num_of_relays + 1)

    individual = list(np.random.uniform(0, 1, size=(length,)))

    return individual

individual = init_individual(inp.num_of_relays, inp.num_of_sensors)

while True:
    indiv = init_individual(inp.num_of_relays, inp.num_of_sensors)
    loss = constructor.get_loss(indiv)
    if loss < 1e8:
        print(loss)
        break