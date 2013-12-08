# -*- coding: utf-8 -*-
#
# This file is part of srs2d.
#
# srs2d is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# srs2d is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with srs2d. If not, see <http://www.gnu.org/licenses/>.

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "19 Sep 2013"

import os
import sys
import argparse
import random
import logging
import physics
import pyopencl as cl
import solace
import png
import subprocess
import tempfile
import math
import numpy as np
import time

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity",        help="increase output verbosity", action="count")
    parser.add_argument("-q", "--quiet",            help="supress output (except errors)", action="store_true")
    parser.add_argument("--device-type",            help="device type (all, gpu or cpu), default is all", type=str, default='all')
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("--trials",                 help="number of trials, default is 500", type=int, default=500)
    parser.add_argument("--num-robots",             help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("--params",                 help="parameters for the neural network", metavar="ANNPARAMS", type=str)
    parser.add_argument("--targets-distances",      help="list of distances between target areas to be evaluated \
        each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("--targets-angles",         help="list of axis angles where the target areas \
        are located each trial (between 0 and PI), default is [3*pi/4]", type=float, nargs='+', default=[2.356194490192345])
    parser.add_argument("--random-targets",         help="place targets at random position (obeying targets distances)", action="store_true")
    args = parser.parse_args()

    if args.verbosity >= 2:
        __log__.setLevel(logging.DEBUG)
    elif args.verbosity == 1:
        __log__.setLevel(logging.INFO)
    else:
        __log__.setLevel(logging.WARNING)

    if args.quiet:
        __log__.setLevel(logging.ERROR)

    uri = os.environ.get('SOLACE_URI')
    username = os.environ.get('SOLACE_USERNAME')
    password = os.environ.get('SOLACE_PASSWORD')

    try:
        git_version = subprocess.check_output('git describe --tags --long'.split(), stderr=subprocess.STDOUT).replace('\n', '')
    except:
        git_version = None

    if (uri is None) or (username is None) or (password is None):
        raise Exception('Environment variables (SOLACE_URI, SOLACE_USERNAME, SOLACE_PASSWORD) not set!')

    device_type = cl.device_type.ALL
    if args.device_type == 'cpu':
        device_type = cl.device_type.CPU
    elif args.device_type == 'gpu':
        device_type = cl.device_type.GPU

    platform = cl.get_platforms()[0]
    devices = platform.get_devices(device_type=device_type)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(1, {
        'STEPS_TA': args.ta,
        'STEPS_TB': args.tb,
        'NUM_ROBOTS': args.num_robots,
        'TARGETS_DISTANCES': args.targets_distances,
        'TARGETS_ANGLES': args.targets_angles,
        'TRIALS': args.trials,
        'PARAMS': args.params if args.params else '',
        'RANDOM_TARGETS': 1 if args.random_targets else 0
    }, code_version=git_version)

    for run in inst.runs:
        ReEval(context, queue, args).execute(run)

class ReEval(object):
    def __init__(self, context, queue, args):
        self.context = context
        self.queue = queue
        self.args = args

        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=1,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb, random_targets=args.random_targets)

        if args.params is not None:
            params = args.params.decode('hex')
        else:
            params = ''
            for i in xrange(physics.ANN_PARAMS_SIZE):
                params += chr(random.randint(0,255))

        self.ann_params = np.zeros(len(params))
        for i in xrange(len(params)):
            self.ann_params[i] = float(ord(params[i])) / 255

        self.step_count = 0

    def execute(self, run=None):
        __log__.info(' ReEval Starting...')

        if run:
            run.begin()

        trial = 1
        while (trial <= self.args.trials):
            __log__.info('[trial=%d] Evaluating solution...', trial)

            _, filename = tempfile.mkstemp(prefix='reeval_', suffix='.srs')

            f = self.simulator.simulate_and_save(
                filename,
                [ self.ann_params ],
                targets_distance=self.args.targets_distances[ random.randint(0, len(self.args.targets_distances)-1) ],
                targets_angle=self.args.targets_angles[ random.randint(0, len(self.args.targets_angles)-1) ]
            )
            fitness = float(f[0])

            run.upload(filename, 'reeval-%02d-fit-%.4f.srs' % (trial, fitness) )
            os.remove(filename)

            __log__.info('[trial=%d] Solution evaluated, fitness = %.5f', trial, fitness)

            if run:
                run.progress(trial / float(self.args.trials), {
                    'trial': trial,
                    'fitness': fitness
                })

            trial += 1

        if run:
            run.done()

if __name__=="__main__":
    main()
