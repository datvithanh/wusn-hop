import os
import numpy as np
from statistics import stdev
root_dir = 'small'
os.chdir('results-2021/pso')
def type_result(syn):
    ##single hop
    #type 1 2 3
    for i in sorted(syn.keys()):
        if len(i.split('_')) != 3:
            continue
        if 'r50' in i:
            continue
        print(i, *syn[i])
    #type 4
    for i in sorted(syn.keys()):
        if len(i.split('_')) != 3:
            continue
        if 'r50' in i and 'uu' in i:
            print(i, *syn[i])
    ##multi hop
    #type 1 2 3
    for i in sorted(syn.keys()):
        if len(i.split('_')) != 4:
            continue
        if 'r50' in i or '_40' in i:
            continue
        print(i, *syn[i])
    #type 4
    for i in sorted(syn.keys()):
        if len(i.split('_')) != 4:
            continue
        if 'r50' in i and 'uu' in i and '_40' not in i:
            print(i, *syn[i])
    #type 5
    for i in sorted(syn.keys()):
        if len(i.split('_')) != 4:
            continue
        if 'r25' in i and 'uu' in i and '_40' in i:
            print(i, *syn[i])

layer_data = {}

for fn in os.listdir('small/layer'):
    with open(os.path.join('small/layer', fn)) as f:
        lines = [tmp.replace('\n', '') for tmp in f.readlines()]
        hop_file = lines[0]
        res = float(lines[-1].split('\t')[-1])
        if res > 10:
            print(fn)
        if hop_file not in layer_data.keys():
            layer_data[hop_file] = {fn: res}
        else:
            layer_data[hop_file][fn] = res

for i in sorted(layer_data.keys()):
    values = list(layer_data[i].values())
    if 'r25' in i.split('.')[0]:
        print(i.split('.')[0], min(values), np.mean(values), stdev(values))
        
for i in sorted(layer_data.keys()):
    values = list(layer_data[i].values())
    if 'r50' in i.split('.')[0]:
        print(i.split('.')[0], min(values), np.mean(values), stdev(values))

hop_data = {}

for fn in sorted(os.listdir('small/hop')):
    with open(os.path.join('small/hop', fn)) as f:
        lines = [tmp.replace('\n', '') for tmp in f.readlines()]
        hop_file = lines[0]
        res = float(lines[-1].split('\t')[-1])
        if res > 10:
            print(fn)
        if hop_file not in hop_data.keys():
            hop_data[hop_file] = {}
        hop_data[hop_file][fn] = res

for i in sorted(hop_data.keys()):
    values = list(hop_data[i].values())
    if 'r25' in i and '_0' in i:
        print(i.split('.')[0], min(values), np.mean(values), stdev(values))

for i in sorted(hop_data.keys()):
    values = list(hop_data[i].values())
    if 'r50' in i and '_0' in i and 'uu' in i:
        print(i.split('.')[0], min(values), np.mean(values), stdev(values))
        
for i in sorted(hop_data.keys()):
    values = list(hop_data[i].values())
    if 'r25' in i and '_40' in i:
        print(i.split('.')[0], min(values), np.mean(values), stdev(values))