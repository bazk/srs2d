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
import ga
import Queue
import threading

ANN_PARAMS_SIZE = 113

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity",        help="increase output verbosity", action="count")
    parser.add_argument("-q", "--quiet",            help="supress output (except errors)", action="store_true")
    parser.add_argument("--device-type",            help="device type (all, gpu or cpu), default is all", type=str, default='all')
    parser.add_argument("--islands-per-device",     help="number of islands per device, default is 1", type=int, default=1)
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="population size (genomes), default is 120", type=int, default=120)
    parser.add_argument("-d", "--distances",        help="list of distances between target areas to be evaluated each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("--fixed-targets",          help="targets will always be in the same position", action="store_true")
    parser.add_argument("-t", "--trials",           help="number of trials per distance, default is 3", type=int, default=3)
    parser.add_argument("-c", "--pcrossover",       help="probability of crossover, default is 0.9", type=float, default=0.9)
    parser.add_argument("-m", "--pmutation",        help="probability of mutation, default is 0.03", type=float, default=0.03)
    parser.add_argument("-o", "--offspring",        help="number of children each couple of indivuals generate, MUST BE EVEN, default is 6", type=int, default=6)
    parser.add_argument("-e", "--elite-size",       help="size of population elite, default is 24", type=int, default=24)
    parser.add_argument("--migration-rate",         help="proportion of individual of a population that migrate, default is 0.1", type=float, default=0.1)
    parser.add_argument("--migration-freq",         help="frequency of migration (in generations), default is 10", type=int, default=10)
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

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(args.num_runs, {
        'PCROSSOVER': args.pcrossover,
        'PMUTATION': args.pmutation,
        'ELITE_SIZE': args.elite_size,
        'OFFSPRING': args.offspring,
        'MIGRATION_RATE': args.migration_rate,
        'MIGRATION_FREQ': args.migration_freq,
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
        PGA(context, args).execute(run)

class PGA:
    def __init__(self, context, args):
        self.context = context
        self.args = args

        self.archipelago = []
        for device in self.context.devices:
            #n = args.islands_per_device

            n = 2
            if device.type == cl.device_type.GPU:
                n = 6

            for i in xrange(n):
                queue = cl.CommandQueue(context, device)
                self.archipelago.append(ga.GA(self.context, queue, args))

        self.worker_queue = Queue.Queue()

        for i in xrange(len(self.archipelago)):
             t = threading.Thread(target=self.worker)
             t.daemon = True
             t.start()

    def execute(self, run):
        __log__.info(' Parallel GA Starting (archipelago size = %d)...' % len(self.archipelago))

        if run:
            run.begin()

        last_best_fitness = None
        generation = 1
        while (generation <= self.args.num_generations):
            __log__.info('[gen=%d] Evaluating archipelago...', generation)

            # evaluate every island
            for island in self.archipelago:
                self.worker_queue.put(island)
            self.worker_queue.join()

            # find avg and best fitness
            self.avg_fitness = 0
            self.best = None
            for island in self.archipelago:
                self.avg_fitness += island.avg_fitness

                if (self.best is None) or (island.best.fitness > self.best.fitness):
                    self.best = island.best

            self.avg_fitness /= len(self.archipelago)

            # migration
            if (generation % self.args.migration_freq) == 0:
                for i in xrange(len(self.archipelago)):
                    cur = self.archipelago[i]
                    next = self.archipelago[(i+1) % len(self.archipelago)]

                    for n in xrange(int(self.args.population_size * self.args.migration_rate)):
                        next.population.append(cur.population.pop())

            # save partial results

            __log__.info('[gen=%d] Archipelago evaluated, avg_fitness = %.5f, best fitness = %.5f', generation, self.avg_fitness, self.best.fitness)

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

                    fitness = self.archipelago[1].simulator.simulate_and_save(
                        self.args.distances[ random.randint(0, len(self.args.distances)-1) ],
                        [ self.best.genome_decoded for i in xrange(self.args.population_size) ],
                        filename
                    )

                    run.upload(filename, 'run-%02d-new-best-gen-%04d-fit-%.4f.srs' % (run.id, generation, fitness[0]) )
                    os.remove(filename)

            generation += 1

        if run:
            run.done()

    def worker(self):
        while True:
            island = self.worker_queue.get()
            island.step()

            if island.queue.device.type == cl.device_type.GPU:
                __log__.debug('GPU island step finished (avg_step_time: %.2f seconds)', island.avg_step_time)
            elif island.queue.device.type == cl.device_type.CPU:
                __log__.debug('CPU island step finished (avg_step_time: %.2f seconds)', island.avg_step_time)
            else:
                __log__.debug('Unknown device island step finished (avg_step_time: %.2f seconds)', island.avg_step_time)

            self.worker_queue.task_done()

if __name__=="__main__":
    main()
