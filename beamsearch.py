from constructor.nrk import Nrk
from utils.input import WusnInput, WusnConstants
from utils.point import distance

class Layer(Nrk):
     def get_loss(self, father=None, num_child=None):     
        if None in [father, num_child]:
            return float('inf')

        max_energy_consumption = 0

        print(num_child)
        
        for index in range(41):
            if index == 0:
                continue
            e_t = WusnConstants.k_bit * WusnConstants.e_elec + \
                WusnConstants.k_bit * WusnConstants.e_fs * distance(self._points[index], self._points[father[index]])
            e_r = num_child[index] * WusnConstants.k_bit * WusnConstants.e_elec                
            
            max_energy_consumption = max(max_energy_consumption, e_t + e_r)

        return max_energy_consumption    


inp = WusnInput.from_file('./data/layer/ga-dem1_r25_1.json')

nrk = Layer(inp, 14, 6, is_hop=False)

tmp = [25, 34, 27, 9, 37, 1, 27, 9, 0, 34, 19, 26, 29, 37, 30, 24, 37, 13, 29, 27, 30, 0, 1, 19, 24, 34, 24, 25, 25, 14, 13, 9, 14, 13, 14, 30, 29, 26, 1, 19]

for idx, val in enumerate(tmp):
    print(distance(nrk._points[idx + 41], nrk._points[val+1]))

rl = [0] * 41
ft = [0] * 41

for i in tmp:
    rl[i] += 1

print(nrk.get_loss(ft, rl))
