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
__date__ = "04 Jul 2013"

import os
import argparse
import random
import logging
import physics
import pyopencl as cl
import logging.config
import solace
import io
import png
import subprocess
import numpy as np
import math
import tempfile

ANN_PARAMS_SIZE = 113

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity",        help="increase output verbosity", action="count")
    parser.add_argument("-q", "--quiet",            help="supress output (except errors)", action="store_true")
    parser.add_argument("--device-type",            help="device type (all, gpu or cpu), default is all", type=str, default='all')
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("-w", "--inertia",          help="set PSO inertia (W) parameter, default is 0.9", type=float, default=0.9)
    parser.add_argument("-a", "--alfa",             help="set PSO alfa parameter, default is 2.0", type=float, default=2)
    parser.add_argument("-b", "--beta",             help="set PSO beta parameter, default is 2.0", type=float, default=2)
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="PSO population size (particles), default is 10", type=int, default=10)
    parser.add_argument("-d", "--distances",        help="list of distances between target areas to be evaluated each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("-t", "--trials",           help="number of trials per distance, default is 3", type=int, default=3)
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
    inst = exp.create_instance(args.num_runs, {
        'W': args.inertia,
        'ALFA': args.alfa,
        'BETA': args.beta,
        'STEPS_TA': args.ta,
        'STEPS_TB': args.tb,
        'NUM_GENERATIONS': args.num_generations,
        'NUM_RUNS': args.num_runs,
        'NUM_ROBOTS': args.num_robots,
        'POPULATION_SIZE': args.population_size,
        'D': args.distances,
        'TRIALS': args.trials,
    }, code_version=git_version)

    for run in inst.runs:
        DiscretePSO(context, queue).execute(run, args)

class DiscretePSO(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def execute(self, run, args):
        __log__.info('Starting DiscretePSO...')

        run.begin()

        self.gbest = None
        self.gbest_fitness = None

        self.particles = [ Particle(ANN_PARAMS_SIZE, args.inertia, args.alfa, args.beta) for i in range(args.population_size) ]

        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=args.population_size,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb)

        generation = 1
        while (generation <= args.num_generations):
            __log__.info('[gen=%d] Evaluating particles...', generation)
            self.evaluate(args.distances, args.trials)

            __log__.info('[gen=%d] Updating particles...', generation)
            for p in self.particles:
                p.update_pbest()

            new_gbest = False
            for p in self.particles:
                if (self.gbest is None) or (p.pbest.fitness > self.gbest.fitness):
                    self.gbest = p.pbest.copy()
                    new_gbest = True

            for p in self.particles:
                p.gbest = self.gbest
                p.update_pos_vel()

            avg_pbest = 0
            for p in self.particles:
                avg_pbest += p.pbest.fitness
            avg_pbest /= len(self.particles)

            __log__.info('[gen=%d] Particles updated, avg(pbest fitness) = %.5f, gbest fitness: %.5f', generation, avg_pbest, self.gbest.fitness)

            run.progress(generation / float(args.num_generations), {
                'generation': generation,
                'avg_pbest_fitness': avg_pbest,
                'gbest_fitness': self.gbest.fitness,
                'gbest_position': self.gbest.position_hex
            })

            if new_gbest and (not args.no_save):
                __log__.info('[gen=%d] Saving simulation for the new found gbest...', generation)
                _, filename = tempfile.mkstemp(prefix='bpso_sim_', suffix='.srs')

                fitness = self.simulator.simulate_and_save(
                    args.distances[ random.randint(0, len(args.distances)-1) ],
                    [ self.gbest.position for i in xrange(len(self.particles)) ],
                    filename
                )

                run.upload(filename, 'run-%02d-new-gbest-gen-%04d-fit-%.4f.srs' % (run.id, generation, fitness[0]) )
                os.remove(filename)

            generation += 1

        run.done()

    def evaluate(self, distances, trials):
        for p in self.particles:
            p.fitness = 0.0

        for d in distances:
            for t in range(trials):
                fitness = self.simulator.simulate(d, [ p.position for p in self.particles ])

                for i in xrange(len(self.particles)):
                    self.particles[i].fitness += fitness[i]

        for p in self.particles:
            p.fitness /= len(distances) * trials

    def generate_image(self, filename, block_width=8, block_height=8):
        blocks = [ [] for p in xrange(len(self.particles)) ]
        pixels = []

        for p in xrange(len(self.particles)):
            pos = self.particles[p].position

            for c in pos:
                blocks[p].append(ord(c))

        for i in xrange(len(blocks[0])):
            line = []
            for b in blocks:
                for x in xrange(block_width):
                    line.append(b[i])

            for y in xrange(block_height):
                pixels.append(line)

        png.from_array(pixels, 'L').save(filename)

class Particle(object):
    MIN = -6
    MAX =  6

    def __init__(self, size, inertia=0.9, alfa=2.0, beta=2.0):
        self.id = id(self)

        self.fitness = 0

        self.inertia = inertia
        self.alfa = alfa
        self.beta = beta

        self.position = ''
        for i in xrange(size):
            self.position += chr(random.randint(0,255))

        self.probabilities = np.random.uniform(self.MIN, self.MAX, (size,255))
        self.velocity = np.random.uniform(self.MIN, self.MAX, (size,255))

        self.pbest = None
        self.gbest = None

    def __repr__(self):
        return 'Particle(%d, fitness=%.5f)' % (self.id, self.fitness)

    @property
    def position_hex(self):
        ret = ''
        for c in self.position:
            ret += c.encode('hex')
        return ret

    def copy(self):
        p = Particle(len(self.position), self.inertia, self.alfa, self.beta)
        p.fitness = self.fitness
        p.position = self.position
        p.velocity = np.copy(self.velocity)
        p.probabilities = np.copy(self.probabilities)
        p.pbest = self.pbest
        p.gbest = self.gbest
        return p

    def update_pbest(self):
        if (self.pbest is None) or (self.fitness > self.pbest.fitness):
            self.pbest = self.copy()

    def update_pos_vel(self):
        self.velocity = self.inertia * self.velocity + \
                        self.alfa * random.uniform(0, 1.0) * (self.pbest.probabilities - self.probabilities) + \
                        self.beta * random.uniform(0, 1.0) * (self.gbest.probabilities - self.probabilities)
        self.velocity = np.clip(self.velocity, self.MIN, self.MAX)

        self.probabilities = self.probabilities + self.velocity
        self.probabilities = np.clip(self.probabilities, self.MIN, self.MAX)

        new_pos = ''
        for i in xrange(len(self.position)):
            p = np.zeros(255)
            for j in xrange(len(self.probabilities[i])):
                p[j] = self.sigmoid(self.probabilities[i][j])
            p /= np.sum(p)

            # weighted choice
            # choices = sorted([ (i, p[i]) for i in xrange(len(p)) ], key=lambda (v, w): w)
            choices = [ (i, p[i]) for i in xrange(len(p)) ]
            r = random.random()
            s = 0
            for v,w in choices:
                if (s+w) > r:
                    break
                s += w

            new_pos += chr(v)

        self.position = new_pos

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

if __name__=="__main__":
    main()