#!/usr/bin/env python

'''
This is a utility tool for collecting performance data.

Created on 4/4/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations

import argparse
import json
import os
import numpy as np

import utils.utils as utils
from fold_sim.wrappers.normalized_reward import SCALE_COEFF

if __name__ == '__main__':
    # Parse arguments
    class Args(argparse.Namespace):
        config_file: str
        out_file: str
        eval_dir: str
        rescale: float
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        help='Configuration JSON file used for training.')
    parser.add_argument('out_file', type=str,
                        help='Output file for collected results.')
    parser.add_argument('--eval-dir', default='model_evaluations',
                        help='Directory where model evaluation checkpoints are stored ("%(default)s" by default).', metavar='DIRNAME')
    parser.add_argument('-r', '--rescale', type=float, default=0.0,
                        help='If provided, rescale normalized rewards back to their unnormalized values with the provided expected input size.')
    args = parser.parse_args(namespace=Args)

    # Use the provided configuration file
    with open(args.config_file) as f:
        config = json.load(f)
    
    # Collect results
    iteration = 0
    rewards = []
    eval_file = utils.eval_file_name(args.eval_dir, config['name'], iteration)
    while os.path.exists(eval_file):
        rewards.append(utils.load_eval(eval_file)['latest_avg_reward'])
        iteration += 1
        eval_file = utils.eval_file_name(args.eval_dir, config['name'], iteration)
    rewards = np.asarray(rewards)
    if args.rescale > 0:
        rewards *= args.rescale / SCALE_COEFF

    # Export results
    np.save(args.out_file, rewards)
