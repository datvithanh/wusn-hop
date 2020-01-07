import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

from collections import deque
from utils.input import WusnInput, WusnConstants
from utils.point import distance

class Hop:
    def __init__(self, inp: WusnInput, max_relay=10, hop=6):
        self._sensors = inp.sensors
        self._relays = inp.relays
        self._radius = inp.radius
        self._num_of_sensors = inp.num_of_sensors
        self._num_of_relays = inp.num_of_relays

        self._max_relay = max_relay
        self._hop = hop

        self.graph_construct(inp)

    def graph_construct(self, inp: WusnInput):
        point2idx = {}
        points = []
        point2idx[inp.BS] = 0
        points.append(inp.BS)
        for i, rl in enumerate(inp.relays):
            point2idx[rl] = i + 1
            points.append(rl)
        for i, sn in enumerate(inp.sensors):
            point2idx[sn] = i + 1 + inp.num_of_relays
            points.append(sn)

        # Construct edge set
        edges = [[] for _ in range(len(points))]

        for i in range(inp.num_of_relays):
            edges[0].append(i)
            edges[i].append(0)
            
        for rl in inp.relays:
            for sn in inp.sensors:
                if distance(rl, sn) <= inp.radius:
                    u, v = point2idx[rl], point2idx[sn]
                    edges[u].append(v)
                    edges[v].append(u)

        for sn1 in inp.sensors:
            for sn2 in inp.sensors:
                if distance(sn1, sn2) <= inp.radius:
                    u, v = point2idx[sn1], point2idx[sn2]
                    if u == v:
                        continue
                    edges[u].append(v)
                    edges[v].append(u)

        self._points = points
        self._point2idx = point2idx
        self._edges = edges
        
    def get_loss(self, individual):
        if sum(individual) > self._max_relay:
            return float("inf")

        hop_count = [0]*(1 + self._num_of_relays + self._num_of_sensors)
        hop_count[0] = 1

        father = [0]*(1 + self._num_of_relays + self._num_of_sensors)

        num_child = [0] * (1 + self._num_of_relays + self._num_of_sensors)


        # Use BFS to find max hop in the network
        queue = deque()
        for index, value in enumerate(individual):
            if value == 1:
                hop_count[index+1] = 2
                father[index+1] = 0
                queue.append(index + 1)

        while len(queue) != 0:
            u = queue.popleft()
            for v in self._edges[u]:
                if v <= self._num_of_relays:
                    continue
                if hop_count[v] == 0 or hop_count[v] > hop_count[u] + 1:
                    hop_count[v] = hop_count[u] + 1
                    father[v] = u
                    queue.append(v)

        # exceed hop count
        if max(hop_count) - 1 > self._hop:
            return float("inf")

        # not a tree
        if min(hop_count[1 + self._num_of_relays:]) == 0:
            # print(hop_count)
            return float("inf")

        # TODO: child balancing within tree
        for depth in range(self._hop, 1, -1):
            for index, value in enumerate(hop_count):
                if value == depth:
                    u = index
                    while u!=0:
                        u = father[u]
                        num_child[u] += 1

        max_energy_consumption = 0

        for index, value in enumerate(hop_count):
            if value > 0:
                if index == 0:
                    continue
                e_t = WusnConstants.k_bit * WusnConstants.e_elec + \
                    WusnConstants.k_bit * WusnConstants.e_fs * distance(self._points[index], self._points[father[index]])
                e_r = num_child[index] * WusnConstants.k_bit * WusnConstants.e_elec                
                
                max_energy_consumption = max(max_energy_consumption, e_t + e_r)
        
        return max_energy_consumption   

class Layer:
    def __init__(self, inp: WusnInput, max_relay):
        self.inp = inp

        self.connection = {}
        self.emax = 0
        self._max_relay = max_relay

        self.gen_connection_matrix()
        self.calculate_emax()

    def gen_connection_matrix(self):
        for sn in self.inp.sensors:
            for rl in self.inp.relays:
                if distance(sn, rl) <= self.inp.radius:
                    self.connection[(sn, rl)] = 1
    
    def calculate_emax(self):
        # num_sensors_to_relay = [0] * self.inp.num_of_relays
        for sn in self.inp.sensors:
            for rn in self.inp.relays:
                if (sn, rn) in self.connection:
                    self.emax = max(self.emax, WusnConstants.k_bit * (WusnConstants.e_elec + \
                            WusnConstants.e_fs*distance(sn, rn)**2) )
        
        for rn in self.inp.relays:
            self.emax = max(self.emax, WusnConstants.k_bit* (WusnConstants.e_elec + \
                WusnConstants.e_fs * distance(rn, self.inp.BS)**2 + \
                self.inp.num_of_sensors * WusnConstants.e_elec))

    def get_loss(self, individual: list): 
        if sum(individual) > self._max_relay:
            return float('inf')
        
        num_sensors_to_relay = [0] * len(individual)
        config = {}
        max_energy_consumption = -float('inf')

        for sid in range(self.inp.num_of_sensors):
            min_max = float("inf")
            selected_id = -1
            for rid, val in enumerate(individual):
                if val == 1:
                    if (self.inp.sensors[sid], self.inp.relays[rid]) in self.connection:
                        sensor_loss = WusnConstants.k_bit * (WusnConstants.e_elec + \
                            WusnConstants.e_fs*distance(self.inp.sensors[sid], self.inp.relays[rid])**2)
                        relay_loss = WusnConstants.k_bit* (WusnConstants.e_elec + \
                            WusnConstants.e_fs * distance(self.inp.relays[rid], self.inp.BS)**2 + \
                            (num_sensors_to_relay[rid] + 1) * WusnConstants.e_elec )

                        if max(sensor_loss, relay_loss) < min_max:
                            min_max = max(sensor_loss, relay_loss)
                            selected_id = rid
            config[self.inp.sensors[sid]] = self.inp.relays[selected_id]
            num_sensors_to_relay[selected_id] +=1

            ss_loss = WusnConstants.k_bit * (WusnConstants.e_elec + \
                WusnConstants.e_fs * distance(self.inp.sensors[sid], self.inp.relays[selected_id])**2)
            rl_loss = WusnConstants.k_bit* (WusnConstants.e_elec + \
                WusnConstants.e_fs * distance(self.inp.relays[selected_id], self.inp.BS)**2 + \
                num_sensors_to_relay[selected_id] * WusnConstants.e_elec)

            max_energy_consumption = max(max_energy_consumption, max([ss_loss, rl_loss]))

        return max_energy_consumption 
        # num_activated_relays = sum([tmp > 0 for tmp in num_sensors_to_relay])
        # return 0.5*num_activated_relays / self.inp.num_of_relays + 0.5*max_energy_consumption / self.emax

if __name__ == "__main__":
    inp = WusnInput.from_file('hop-data/hop/ga-dem5_r25_1.json')

    hc = Hop(inp, hop=10)

    print(hc.get_loss([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
        
