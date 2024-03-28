import json
import os
import pprint
import sys
import argparse
import math

sys.path.append("./pSp")

from psp_training_options import TrainOptions
from pSp.training.coach_new import Coach

if __name__ == '__main__':
    device = 'cuda'

    args = TrainOptions().parse()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)

    args.use_spatial_mapping = not args.no_spatial_map

    coach = Coach(args)
    coach.train()

