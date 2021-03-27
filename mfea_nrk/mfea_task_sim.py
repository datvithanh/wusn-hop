import os, sys
import copy
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import random
import numpy as np
import joblib
import time
import argparse
from datetime import datetime

from utils.input import WusnInput
from constructor.binary import Layer
from constructor.nrk import Nrk
from utils.logger import init_log
from utils import config

from mfea_nrk.mfea import run_ga

N_GENS = config.N_GENS
POP_SIZE = config.POP_SIZE
CXPB = config.MFEA_CXPB
TERMINATE = 30
num_of_relays = config.MAX_RELAYS
num_hops = config.MAX_HOPS

def solve(fns, pas, logger=None):
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}] solving {fns} pas {pas}')

    flog = open(f"results/rmp{CXPB}/mfea{sum(['layer' in tmp for tmp in fns])}{sum(['hop' in tmp for tmp in fns])}/{pas}", 'w+')
    # flog = open(f"results/mfea{sum(['layer' in tmp for tmp in fns])}{sum(['hop' in tmp for tmp in fns])}/{pas}", 'w+')

    flog.write(f'{fns}\n')

    while not run_ga(fns, flog, logger):
        flog = open(f"results/rmp{CXPB}/mfea{sum(['layer' in tmp for tmp in fns])}{sum(['hop' in tmp for tmp in fns])}/{pas}", 'w+')
        # flog = open(f"results/mfea{sum(['layer' in tmp for tmp in fns])}{sum(['hop' in tmp for tmp in fns])}/{pas}", 'w+')
        flog.write(f'{fns}\n')

    print(f'done solved {fns[1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rmp', type=float, default=0.1)
    parser.add_argument('--task-file', type=str, default='data/tasks/run_total.txt', help='path to task file')
    parser.add_argument('--n-jobs', type=int, default=4)

    args = parser.parse_args()

    CXPB = args.rmp

    logger = init_log()
    os.makedirs(f'results/rmp{CXPB}/mfea11', exist_ok=True)
    os.makedirs(f'results/rmp{CXPB}/mfea31', exist_ok=True)
    os.makedirs(f'results/rmp{CXPB}/mfea13', exist_ok=True)
    os.makedirs(f'results/rmp{CXPB}/mfea33', exist_ok=True)
    # os.makedirs(f'results/mfea11', exist_ok=True)
    # os.makedirs(f'results/mfea31', exist_ok=True)
    # os.makedirs(f'results/mfea13', exist_ok=True)
    # os.makedirs(f'results/mfea33', exist_ok=True)

    lines = [tmp.replace('\n', '') for tmp in open(args.task_file, 'r').readlines()]

    tests, pases = zip(*[tmp.split('\t') for tmp in lines])
    tests = [tmp.split(' ') for tmp in tests]

    print(len(tests))
    print(len(pases))
    tests = tests[::-1]
    pases = pases[::-1]
    
    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(solve)(fn, pas=pas, logger=logger) for fn, pas in zip(tests, pases)
    )
    # for fn, pas in zip(tests, pases):
    #     solve(fn, pas=pas, logger=logger)
