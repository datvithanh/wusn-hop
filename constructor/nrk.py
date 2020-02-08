import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

from collections import deque
from utils.input import WusnInput, WusnConstants
from utils.point import distance

class Nrk:
    def __init__(self, inp: WusnInput, max_relay=10, hop=6, is_hop=True):
        self._sensors = inp.sensors
        self._relays = inp.relays
        self._radius = inp.radius
        self._num_of_sensors = inp.num_of_sensors
        self._num_of_relays = inp.num_of_relays

        self._max_relay = max_relay
        self._hop = hop
        self._is_hop = is_hop

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
        if self._is_hop == True:
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

    def transform_genes(self, individual, num_of_sensors, max_rl, max_ss):
        first_half = individual[:1 + max_rl + num_of_sensors]
        second_half = individual[1 + max_rl + max_ss:2 + 2*max_rl + max_ss + num_of_sensors]

        return first_half + second_half
    
    def decode_genes(self, individual):
        # decode tree from genes
        father = [0]*(1 + self._num_of_relays + self._num_of_sensors)

        dict_individual_order = {i: individual[i] for i in range(len(individual) // 2)}
        dict_individual_value = {i - len(individual) // 2: individual[i] for i in range(len(individual) // 2, len(individual))}

        selected_relays = list(
            dict(sorted({i: individual[i] for i in range(1, self._num_of_relays + 1)}.items(), key=lambda x: x[1])).keys())[
                         :self._max_relay]
        
        not_selected_relays = [i for i in range(1, self._num_of_relays + 1) if i not in selected_relays]

        for pos in selected_relays:
            father[pos] = 0
        
        order = list(dict(sorted(list(dict_individual_order.items())[self._num_of_relays+1:], key=lambda x: x[1])).keys())

        for sid in order:
            dict_order_adjacent = {}
            for v in self._edges[sid]:
                if v not in not_selected_relays:
                    dict_order_adjacent[v] = dict_individual_value[v]

            order_adjacent = list(dict(sorted(dict_order_adjacent.items(), key=lambda x: x[1])).keys())
            for v in order_adjacent:
                ok = None
                current = v
                visited = set([sid])
                while True:
                    if current == 0:
                        ok = True
                        break

                    if current in visited:
                        ok = False
                        break

                    visited.add(current)
                    current = father[current]

                if ok == True:
                    father[sid] = v
                    break
        
        # calculate max energy consumption
        hop_count = [0]*(1 + self._num_of_relays + self._num_of_sensors)
        hop_count[0] = 1
        num_child = [0] * (1 + self._num_of_relays + self._num_of_sensors)

        queue = deque()
        for r in selected_relays:
            hop_count[r] = 2
            queue.append(r)

        while len(queue) != 0:
            u = queue.popleft()
            for v in self._edges[u]:
                if v <= self._num_of_relays:
                    continue
                if hop_count[v] == 0 and father[v] == u:
                    hop_count[v] = hop_count[u] + 1
                    father[v] = u
                    queue.append(v)

        # not a tree
        if min(hop_count[1 + self._num_of_relays:]) == 0:
            return None, None, None

        # exceed hop count
        if max(hop_count) - 1 > self._hop:
            return None, None, None

        for depth in range(self._hop, 1, -1):
            for index, value in enumerate(hop_count):
                if value == depth:
                    u = index
                    while u!=0:
                        u = father[u]
                        num_child[u] += 1

        return father, num_child, hop_count


    def get_loss(self, individual):
        father, num_child, hop_count = self.decode_genes(individual)
     
        if None in [father, num_child]:
            return float('inf')

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