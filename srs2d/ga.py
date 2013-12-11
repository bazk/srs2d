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
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="population size (genomes), default is 120", type=int, default=120)
    parser.add_argument("--targets-distances",      help="list of distances between target areas to be evaluated \
        each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("--targets-angles",         help="list of axis angles where the target areas \
        are located each trial (between 0 and PI), default is [3*pi/4]", type=float, nargs='+', default=[2.356194490192345])
    parser.add_argument("--random-targets",         help="place targets at random position (obeying targets distances)", action="store_true")
    parser.add_argument("--symetrical-targets",     help="place targets at symetrical position", action="store_true")
    parser.add_argument("-t", "--trials",           help="number of trials per distance, default is 3", type=int, default=3)
    parser.add_argument("-c", "--pcrossover",       help="probability of crossover, default is 0.9", type=float, default=0.9)
    parser.add_argument("-m", "--pmutation",        help="probability of mutation, default is 0.03", type=float, default=0.03)
    parser.add_argument("-o", "--offspring",        help="number of children each couple of indivuals generate, MUST BE EVEN, default is 6", type=int, default=6)
    parser.add_argument("-e", "--elite-size",       help="size of population elite, default is 24", type=int, default=24)
    args = parser.parse_args()

    if args.verbosity >= 2:
        __log__.setLevel(logging.DEBUG)
    elif args.verbosity == 1:
        __log__.setLevel(logging.INFO)
    else:
        __log__.setLevel(logging.WARNING)

    if args.quiet:
        __log__.setLevel(logging.ERROR)

    if (args.offspring % 2) != 0:
        __log__.error('Offspring must be an even number!')
        sys.exit(1)

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
        'PCROSSOVER': args.pcrossover,
        'PMUTATION': args.pmutation,
        'ELITE_SIZE': args.elite_size,
        'OFFSPRING': args.offspring,
        'STEPS_TA': args.ta,
        'STEPS_TB': args.tb,
        'NUM_GENERATIONS': args.num_generations,
        'NUM_RUNS': args.num_runs,
        'NUM_ROBOTS': args.num_robots,
        'POPULATION_SIZE': args.population_size,
        'TARGETS_DISTANCES': args.targets_distances,
        'TARGETS_ANGLES': args.targets_angles,
        'TRIALS': args.trials,
        'RANDOM_TARGETS': 1 if args.random_targets else 0,
        'SYMETRICAL_TARGETS': 1 if args.symetrical_targets else 0
    }, code_version=git_version)

    for run in inst.runs:
        GA(context, queue, args).execute(run)

class GA(object):
    def __init__(self, context, queue, args):
        self.context = context
        self.queue = queue
        self.args = args

        self.population = [ Individual(physics.ANN_PARAMS_SIZE) for i in range(args.population_size) ]
        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=args.population_size,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb,
                                           random_targets=args.random_targets,
                                           symetrical_targets=args.symetrical_targets)

        self.avg_fitness = None
        self.best = None

        self.step_count = 0
        self.avg_step_time = 0

    def execute(self, run=None):
        __log__.info(' GA Starting...')

        if run:
            run.begin()

        last_best_fitness = None
        generation = 1
        while (generation <= self.args.num_generations):
            __log__.info('[gen=%d] Evaluating population...', generation)

            self.step()

            __log__.info('[gen=%d] Population evaluated, avg_fitness = %.5f, best fitness = %.5f', generation, self.avg_fitness, self.best.fitness)

            if run:
                run.progress(generation / float(self.args.num_generations), {
                    'generation': generation,
                    'avg_fitness': self.avg_fitness,
                    'best_fitness': self.best.fitness,
                    'best_genome': self.best.genome_hex
                })

            if (last_best_fitness is None) or (self.best.fitness > last_best_fitness):
                last_best_fitness = self.best.fitness

                if run and (not self.args.no_save):
                    __log__.info('[gen=%d] Saving simulation for the new found best...', generation)
                    _, filename = tempfile.mkstemp(prefix='sim_', suffix='.srs')

                    fitness = self.simulator.simulate_and_save(
                        filename,
                        [ self.best.genome_decoded for i in xrange(len(self.population)) ],
                        targets_distance=self.args.targets_distances[ random.randint(0, len(self.args.targets_distances)-1) ],
                        targets_angle=self.args.targets_angles[ random.randint(0, len(self.args.targets_angles)-1) ]
                    )

                    run.upload(filename, 'run-%02d-new-best-gen-%04d-fit-%.4f.srs' % (run.id, generation, fitness[0]) )
                    os.remove(filename)

            generation += 1

        if run:
            run.done()

    def step(self):
        start = time.time()

        (self.avg_fitness, self.best) = self.evaluate(self.args.targets_distances, self.args.targets_angles, self.args.trials)

        # Generate new pop
        elite = []
        for i in self.population[-self.args.elite_size:]:
            elite.append(i.copy())

        new_pop = []

        remaining = 0
        if (len(self.population) % self.args.offspring) != 0:
            remaining = len(self.population) % self.args.offspring

        for i in xrange(len(self.population) / self.args.offspring):
            father = self.select()
            mother = self.select()

            for j in xrange(self.args.offspring / 2):
                if random.random() < self.args.pcrossover:
                    brother, sister = father.crossover(mother)
                else:
                    brother = father.copy()
                    sister = mother.copy()

                brother.mutate(self.args.pmutation)
                sister.mutate(self.args.pmutation)

                new_pop.append(brother)
                new_pop.append(sister)

        if remaining > 0:
            father = self.select()

            for i in xrange(remaining):
                individual = father.copy()
                individual.mutate(self.args.pmutation)
                new_pop.append(individual)

        random.shuffle(new_pop)
        for i in xrange(self.args.elite_size):
            new_pop[i] = elite[i]

        self.population = new_pop

        end = time.time()
        self.step_count += 1
        self.avg_step_time = (self.avg_step_time * (self.step_count - 1) + (end - start)) / self.step_count

    def select(self):
        return self.population.pop()

    def evaluate(self, targets_distances, targets_angles, trials):
        for i in xrange(len(self.population)):
            self.population[i].fitness = 0

        for d in targets_distances:
            for a in targets_angles:
                for t in range(trials):
                    fitness = self.simulator.simulate([ ind.genome_decoded for ind in self.population ], targets_distance=d, targets_angle=a)

                    for i in xrange(len(self.population)):
                        self.population[i].fitness += fitness[i]

        for i in xrange(len(self.population)):
            self.population[i].fitness /= len(targets_distances) * len(targets_angles) * trials

        self.population = sorted(self.population, key=lambda ind: ind.fitness)

        avg_fitness = 0
        for ind in self.population:
            avg_fitness += ind.fitness
        avg_fitness /= len(self.population)

        best = self.population[-1]

        return (avg_fitness, best)

    def generate_image(self, filename, block_width=8, block_height=8):
        blocks = [ [] for p in xrange(len(self.population)) ]
        pixels = []

        for p in xrange(len(self.population)):
            gen = self.population[p].genome

            for c in gen:
                blocks[p].append(ord(c))

        for i in xrange(len(blocks[0])):
            line = []
            for b in blocks:
                for x in xrange(block_width):
                    line.append(b[i])

            for y in xrange(block_height):
                pixels.append(line)

        png.from_array(pixels, 'L').save(filename)


class Individual(object):
    def __init__(self, genome_length):
        self.id = id(self)

        self.fitness = 0

        self.genome = ''
        for i in xrange(genome_length):
            self.genome += chr(random.randint(0,255))

    def __repr__(self):
        return 'Individual(%d, fitness=%.5f)' % (self.id, self.fitness)

    @property
    def genome_hex(self):
        ret = ''
        for c in self.genome:
            ret += c.encode('hex')
        return ret

    @property
    def genome_decoded(self):
        ret = np.zeros(len(self.genome))
        for i in xrange(len(self.genome)):
            ret[i] = float(ord(self.genome[i])) / 255
        return ret

    def copy(self):
        g = Individual(len(self.genome))
        g.fitness = self.fitness
        g.genome = self.genome
        return g

    def crossover(self, other):
       """ Single Point crossover """
       point = random.randint(1, len(self.genome)*8-1)

       sister = self.copy()
       brother = other.copy()

       sister.merge(other, point)
       brother.merge(self, point)

       return (sister, brother)

    def mutate(self, pmutation):
        for i in xrange(len(self.genome) * 8):
            if random.random() < pmutation:
                self.flip(i)

    def merge(self, other, point):
        if len(self.genome) != len(other.genome):
            raise Exception('Cannot merge individuals with different genome lenghts.')

        if (point < 0) or (point > (len(self.genome)*8 - 1)):
            raise Exception('Point out of bounds.')

        idx = int(math.floor(float(point) / 8.0))
        bit = 7 - (point % 8)  # big endian

        new = self.genome[:idx]

        sc = ord(self.genome[idx])
        oc = ord(other.genome[idx])
        r = 0
        for i in range(8):
            if (i < bit):
                r |= sc & (2 ** i)
            else:
                r |= oc & (2 ** i)
        new += chr(r)

        new += other.genome[(idx+1):]

        self.genome = new

    def flip(self, point):
        if (point < 0) or (point > (len(self.genome)*8 - 1)):
            raise Exception('Point out of bounds.')

        idx = int(math.floor(float(point) / 8.0))
        bit = 7 - (point % 8)  # big endian

        c = ord(self.genome[idx])
        if (c & (2**bit) != 0):
            c &= ~ (2**bit)
        else:
            c |= 2**bit

        self.genome = self.genome[:idx] + chr(c) + self.genome[(idx+1):]

if __name__=="__main__":
    main()
