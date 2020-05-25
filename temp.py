import os

lines = [tmp.replace('\n', '') for tmp in open('run.txt', 'r').readlines()]

tests, pases = zip(*[tmp.split('\t') for tmp in lines])
tests = [tmp.split(' ') for tmp in tests]
asd = [(1,1), (1,3), (3,1), (3,3)]
for t in tests:
    if (sum(['layer' in tmp for tmp in t]),sum(['hop' in tmp for tmp in t])) not in asd:
        print(t)