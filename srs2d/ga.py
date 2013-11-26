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

ANN_PARAMS_SIZE = 113

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity",        help="increase output verbosity", action="count")
    parser.add_argument("-q", "--quiet",            help="supress output (except errors)", action="store_true")
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="population size (genomes), default is 120", type=int, default=120)
    parser.add_argument("-d", "--distances",        help="list of distances between target areas to be evaluated each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
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

    context = cl.create_some_context()
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
        'D': args.distances,
        'TRIALS': args.trials,
    }, code_version=git_version)

    for run in inst.runs:
        GA(context, queue).execute(run, args)

class GA(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def execute(self, run, args):
        __log__.info(' GA Starting...')

        run.begin()

        self.population = [ Individual(ANN_PARAMS_SIZE) for i in range(args.population_size) ]
        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=args.population_size,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb)

        last_best_fitness = None

        generation = 1
        while (generation <= args.num_generations):
            __log__.info('[gen=%d] Evaluating population...', generation)

            (avg_fitness, best) = self.evaluate(args.distances, args.trials)

            __log__.info('[gen=%d] Population evaluated, avg_fitness = %.5f, best fitness = %.5f', generation, avg_fitness, best.fitness)

            run.progress(generation / float(args.num_generations), {
                'generation': generation,
                'avg_fitness': avg_fitness,
                'best_fitness': best.fitness,
                'best_genome': best.genome_hex
            })

            if (last_best_fitness is None) or (best.fitness > last_best_fitness):
                last_best_fitness = best.fitness

                if not args.no_save:
                    __log__.info('[gen=%d] Saving simulation for the new found best...', generation)
                    _, filename = tempfile.mkstemp(prefix='sim_', suffix='.srs')

                    fitness = self.simulator.simulate_and_save(
                        args.distances[ random.randint(0, len(args.distances)-1) ],
                        [ best.genome for i in xrange(len(self.population)) ],
                        filename
                    )

                    run.upload(filename, 'run-%02d-new-best-gen-%04d-fit-%.4f.srs' % (run.id, generation, fitness[0]) )
                    os.remove(filename)

            # Generate new pop
            elite = []
            for i in self.population[-args.elite_size:]:
                elite.append(i.copy())

            new_pop = []

            remaining = 0
            if (len(self.population) % args.offspring) != 0:
                remaining = len(self.population) % args.offspring

            for i in xrange(len(self.population) / args.offspring):
                father = self.select()
                mother = self.select()

                for j in xrange(args.offspring / 2):
                    if random.random() < args.pcrossover:
                        brother, sister = father.crossover(mother)
                    else:
                        brother = father.copy()
                        sister = mother.copy()

                    brother.mutate(args.pmutation)
                    sister.mutate(args.pmutation)

                    new_pop.append(brother)
                    new_pop.append(sister)

            if remaining > 0:
                father = self.select()

                for i in xrange(remaining):
                    individual = father.copy()
                    individual.mutate(args.pmutation)
                    new_pop.append(individual)

            random.shuffle(new_pop)
            for i in xrange(args.elite_size):
                new_pop[i] = elite[i]

            self.population = new_pop

            generation += 1

        run.done()

    def select(self):
        return self.population.pop()

    def evaluate(self, distances, trials):
        for i in xrange(len(self.population)):
            self.population[i].fitness = 0

        for d in distances:
            for t in range(trials):
                fitness = self.simulator.simulate(d, [ ind.genome for ind in self.population ])

                for i in xrange(len(self.population)):
                    self.population[i].fitness += fitness[i]

        for i in xrange(len(self.population)):
            self.population[i].fitness /= len(distances) * trials

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
