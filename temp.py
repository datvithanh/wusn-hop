import os
dirs = sorted(os.listdir('data/hop'))
examples = []
for fn in dirs: 
    for i in range(5): 
        examples.append(f'{fn[:-5]}_{i}.txt')
        
with open('run_hop.txt', 'w+') as f:
    for i in examples:
        f.write(f'{i}\n')


dirs = sorted(os.listdir('data/layer'))
examples = []
for fn in dirs: 
    for i in range(5): 
        examples.append(f'{fn[:-5]}_{i}.txt')
        
with open('run_layer.txt', 'w+') as f:
    for i in examples:
        f.write(f'{i}\n')