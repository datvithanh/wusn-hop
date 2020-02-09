import os
dirs = sorted(os.listdir('data/hop'))
examples = []
for fn in dirs: 
    for i in range(4): 
        examples.append(f'{fn[:-5]}_{i}.txt')
        
with open('rerunm.txt', 'w+') as f:
    for i in examples:
        f.write(f'{i}\n')