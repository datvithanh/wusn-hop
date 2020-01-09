import os
import sys
sys.path.append('.')

from logzero import logger
from argparse import ArgumentParser
from scipy import interpolate
import numpy as np

from utils.point import Point, SensorNode, RelayNode, distance
from utils.dems_input import DemsInput
from utils.input import WusnInput


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='dems', nargs='+')
    parser.add_argument('-o', '--output', default='data/small_data')
    parser.add_argument('-W', type=int, default=200)
    parser.add_argument('-H', type=int, default=200)
    parser.add_argument('--rows', type=int, default=41)
    parser.add_argument('--cols', type=int, default=41)
    parser.add_argument('--csize', type=int, default=25)
    parser.add_argument('-n', '--count', type=int, default=1)
    parser.add_argument('--depth', type=float, default=1)
    parser.add_argument('--height', type=float, default=10)
    parser.add_argument('--ns', '--num-sensor', type=int, default=40)
    parser.add_argument('--nr', '--num-relay', type=int, default=40)
    parser.add_argument('--radius', default='25,30,45,50')
    parser.add_argument('--prefix', default='uu-')
    parser.add_argument('--distribution', type=str, default="uniform")
    return parser.parse_args()


def point(dinp, xrange, yrange, z_off=0, cls=Point, distribution="uniform"):
    epsilon = 1e-3
    #distribution = {uniform, gaussian, gamma}
    if distribution == "uniform":
        x = np.random.uniform(*xrange)
        y = np.random.uniform(*yrange)
    
    if distribution == "gaussian":
        x = np.clip(np.random.normal(xrange[1]/2, xrange[1]/4), xrange[0], xrange[1]-epsilon)
        y = np.clip(np.random.normal(yrange[1]/2, yrange[1]/4), yrange[0], yrange[1]-epsilon)

    if distribution == "gamma":
        x = np.clip(np.random.gamma(2, 1.5)*xrange[1]/10, xrange[0], xrange[1]-epsilon)
        y = np.clip(np.random.gamma(2, 1.5)*yrange[1]/10, yrange[0], yrange[1]-epsilon)

    z = estimate(x, y, dinp) + z_off
    return cls(x, y, z)


def estimate(x, y, inp: DemsInput):
    id1 = int(x // inp.cellsize)
    id2 = int(y // inp.cellsize)
    xx = [id1 * inp.cellsize, (id1+1) * inp.cellsize]
    yy = [id2 * inp.cellsize, (id2+1) * inp.cellsize]
    z = [[inp.height[id1][id2], inp.height[id1][id2+1]], [inp.height[id1+1][id2], inp.height[id1+1][id2+1]]]
    plane = interpolate.interp2d(xx, yy, z)
    return plane(x, y)[0]


def is_covered(sn, bs_, relays_, radius):
    if distance(sn, bs_) <= radius:
        return False
    for rn_ in relays_:
        if distance(sn, rn_) <= radius:
            return True
    return False

def hop_is_covered(sn, bs_, relays_, sensors_, radius):
    if distance(sn, bs_) <= radius:
        return False
    
    for rn_ in relays_:
        if distance(sn, rn_) <= radius:
            return False
    
    for sn_ in sensors_:
        if distance(sn, sn_) <= radius:
            return True
    
    return False

if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    print(args.distribution)
    radi = list(map(lambda x: int(x), args.radius.split(',')))
    for inp_ in args.dems:
        dem = DemsInput.from_file(inp_)
        dem.scale(args.cols, args.rows, args.csize)
        dname = os.path.split(inp_)[-1].split('.')[0]
        for r in radi:
            for i in range(args.count):
                if not (args.prefix == 'uu-' and dname == 'dem2' and r == 25):
                    break

                fname = '%s%s_r%d_%d.json' % (args.prefix, dname, r, i+1)
                
                fpath = os.path.join(args.output, 'layer', fname)

                # if os.path.exists(fpath):
                #     continue
                logger.info('Generating %s' % fpath)

                # generate random bs
                center_x, center_y = np.random.uniform(args.W/5, args.W - args.W/5), np.random.uniform(args.W/5, args.W - args.W/5)
                center_z = estimate(center_x, center_y, dem)
                bs = Point(center_x, center_y, center_z)

                # Generate random relays
                relays = []

                for j in range(args.nr):
                    rn = None
                    while True:
                        rn = point(dem, (0, args.W), (0, args.H), z_off=args.height, cls=RelayNode, distribution=args.distribution)
                        if distance(rn, bs) <= r:
                            break
                    if distance(rn, bs) > r:
                        logger.warning(f'{r} {distance(rn, bs)}')
                    relays.append(rn)

                # Generate random sensors
                sensors = []
                for j in range(args.ns):
                    ok, sn = False, None
                    while not ok:
                        sn = point(dem, (0, args.W), (0, args.H), z_off=-args.depth, cls=SensorNode, distribution=args.distribution)
                        ok = is_covered(sn, bs, relays, r)
                    sensors.append(sn)

                res = WusnInput(_W=args.W, _H=args.H, _depth=args.depth,
                                _height=args.height, _num_of_relays=args.nr,
                                _num_of_sensors=args.ns, _relays=relays,
                                _sensors=sensors, _BS=bs, _radius=r)

                fpath = os.path.join(args.output, 'layer', fname)
                res.to_file(fpath)

                fpath = os.path.join(args.output, 'hop', '%s%s_r%d_%d_0.json' % (args.prefix, dname, r, i+1))
                res.to_file(fpath)

                for j in range(args.ns):
                    ok, sn = False, None
                    while True:
                        sn = point(dem, (0, args.W), (0, args.H), z_off=-args.depth, cls=SensorNode, distribution=args.distribution)
                        if hop_is_covered(sn, bs, relays, sensors, r):
                            break
                    sensors.append(sn)
                
                res = WusnInput(_W=args.W, _H=args.H, _depth=args.depth,
                                _height=args.height, _num_of_relays=args.nr,
                                _num_of_sensors=len(sensors), _relays=relays,
                                _sensors=sensors, _BS=bs, _radius=r)

                fpath = os.path.join(args.output, 'hop', '%s%s_r%d_%d_40.json' % (args.prefix, dname, r, i+1))
                res.to_file(fpath)

                
    logger.info('Done')
