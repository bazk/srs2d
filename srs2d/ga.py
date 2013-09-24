# -*- coding: utf-8 -*-
#
# This file is part of srs2d.
#
# srs2d is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# trooper-simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with srs2d. If not, see <http://www.gnu.org/licenses/>.

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "19 Sep 2013"

import os
import random
import logging
import physics
import pyopencl as cl
import logging.config
import logconfig
import solace
import io

logging.config.dictConfig(logconfig.LOGGING)

__log__ = logging.getLogger(__name__)

NUM_SENSORS     = 13
NUM_ACTUATORS   = 4
NUM_HIDDEN      = 3

STEPS_TA = 18600
STEPS_TB = 5400

NUM_GENERATIONS = 1000
NUM_RUNS = 1
NUM_ROBOTS = 10
POPULATION_SIZE = 120

D = [0.7, 0.9, 1.1, 1.3, 1.5]

PCROSSOVER = 0.9
PMUTATION = 0.02

class GA(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def select(self):
        return heapq.heappop(self.population)

    def evaluate(self):
        for i in xrange(len(self.population)):
            params = self.population[i][1].decode()
            self.simulator.set_ann_parameters(i, params)
        self.simulator.commit_ann_parameters()

        for i in xrange(len(self.population)):
            self.population[i][0] = .0

        for d in D:
            for i in range(3):
                self.simulator.init_worlds(d)
                self.simulator.simulate()

                fit = self.simulator.get_fitness()
                for i in xrange(len(self.population)):
                    self.population[i][0] += fit[i]

        for i in xrange(len(self.population)):
            self.population[i][0] /= len(D) * 3

    def execute(self, run):
        __log__.info(' GA Starting...')
        __log__.info('=' * 80)

        run.begin()

        self.population = [ (0, Genome()) for i in range(POPULATION_SIZE) ]
        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=POPULATION_SIZE,
                                           num_robots=NUM_ROBOTS,
                                           ta=STEPS_TA, tb=STEPS_TB)

        __log__.info('Calculating fitness for each individual...')
        self.evaluate()

        generation = 0
        while (generation < NUM_GENERATIONS):
            genomeMom = None
            genomeDad = None

            new_pop = []

            size = len(self.population)
            if (size % 2) != 0:
                size -= 1

            for i in xrange(0, , 2):
                genomeMom = self.select()
                genomeDad = self.select()

                if (PCROSSOVER >= 1) or (random.random() < PCROSSOVER):
                   (sister, brother) = genomeMom.crossover(genomeDad)

                if (PMUTATION >= 1) or (random.random() < PMUTATION):
                    sister.mutate()

                if (PMUTATION >= 1) or (random.random() < PMUTATION):
                    brother.mutate()

                new_pop.append(sister)
                new_pop.append(brother)

            if len(self.population) % 2 != 0:
                last = self.select()

                if (PMUTATION >= 1) or (random.random() < PMUTATION):
                    last.mutate()

                new_pop.append(last)

            self.population = new_pop

            __log__.info('Calculating fitness for each individual...')
            self.evaluate()

            generation += 1

            best_genome = max(self.population)[0]
            new_best = False
            if (best_genome[0] > last_best_fitness):
                __log__.info('Found new best genome: %f', best_genome[0])
                new_best = True

            last_best_fitness = best_genome[0]

            __log__.info('-' * 80)
            __log__.info('[gen=%d] CURRENT BEST GENOME IS: %f', generation, best_genome[0])
            __log__.info(str(best_genome[1]))
            __log__.info('-' * 80)

            run.progress(generation / float(NUM_GENERATIONS), {'best_genome_fitness': best_genome[0], 'best_genome': best_genome[1].decode().to_dict()})

            # if new_best:
            #     __log__.info('Saving simulation for the new found best genome...')
            #     self.simulate_and_save('/tmp/simulation.srs', self.gbest.position, D[3])
            #     run.upload('/tmp/simulation.srs', 'run-%d-new-gbest-gen-%d.srs' % (run.id, generation) )

        run.done({'best_genome_fitness': best_genome[0], 'best_genome': best_genome[1].decode().to_dict()})

    # def simulate_and_save(self, filename, pos, distance):
    #     simulator = physics.Simulator(self.context, self.queue,
    #                                   num_worlds=1,
    #                                   num_robots=NUM_ROBOTS,
    #                                   ta=STEPS_TA, tb=STEPS_TB)

    #     save = io.SaveFile.new(filename, step_rate=1/float(simulator.time_step))

    #     simulator.set_ann_parameters(0, pos)
    #     simulator.commit_ann_parameters()
    #     simulator.init_worlds(distance)

    #     arena, target_areas, target_areas_radius = simulator.get_world_transforms()
    #     save.add_square(.0, .0, arena[0][0], arena[0][1])
    #     save.add_circle(target_areas[0][0], target_areas[0][1], target_areas_radius[0][0], .0, .1)
    #     save.add_circle(target_areas[0][2], target_areas[0][3], target_areas_radius[0][1], .0, .1)

    #     transforms, radius = simulator.get_transforms()
    #     robot_radius = radius[0][0]

    #     robot_obj = [ None for i in range(len(transforms)) ]
    #     for i in range(len(transforms)):
    #         robot_obj[i] = save.add_circle(transforms[i][0], transforms[i][1], robot_radius, transforms[i][2], transforms[i][3])

    #     max_steps = STEPS_TA + STEPS_TB
    #     current_step = 0
    #     while current_step < max_steps:
    #         simulator.step()
    #         transforms, radius = simulator.get_transforms()
    #         for i in range(len(transforms)):
    #             robot_obj[i].update(transforms[i][0], transforms[i][1], robot_radius, transforms[i][2], transforms[i][3])
    #         save.frame()
    #         current_step += 1

    #     save.close()

class Genome(object):
    def __init__(self):
        self.id = id(self)

        self.internal = []

        self.length = NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN) + (NUM_ACTUATORS) + \
                        (NUM_HIDDEN * NUM_SENSORS) + NUM_HIDDEN + NUM_HIDDEN

        self.lengths = {
            'weights': NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN),
            'bias': NUM_ACTUATORS,
            'weights_hidden': NUM_HIDDEN * NUM_SENSORS,
            'bias_hidden': NUM_HIDDEN,
            'timec_hidden': NUM_HIDDEN,
        }

    def __str__(self):
        return 'Genome(%d)' % (self.id)

    def copy(self):
        g = Genome()
        g.internal = self.internal.copy()
        return g

    def crossover(self, other):
       """ Single Point crossover """

       if len(self.internal) == 1:
          raise Exception('Cannot crossover genome of length 1.')

       point = random.randint(1, len(self.internal)-1)

       sister = self.copy()
       sister.internal.merge(point, other.internal)

       brother = other.copy()
       brother.internal.merge(point, self.internal)

       return (sister, brother)

    def decode(self):
        return self.internal

if __name__=="__main__":
    uri = os.environ.get('SOLACE_URI')
    username = os.environ.get('SOLACE_USERNAME')
    password = os.environ.get('SOLACE_PASSWORD')

    if (uri is None) or (username is None) or (password is None):
        raise Exception('Environment variables (SOLACE_URI, SOLACE_USERNAME, SOLACE_PASSWORD) not set!')

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(NUM_RUNS, {
        'NUM_SENSORS': NUM_SENSORS,
        'NUM_ACTUATORS': NUM_ACTUATORS,
        'NUM_HIDDEN': NUM_HIDDEN,
        'STEPS_TA': STEPS_TA,
        'STEPS_TB': STEPS_TB,
        'NUM_GENERATIONS': NUM_GENERATIONS,
        'NUM_RUNS': NUM_RUNS,
        'NUM_ROBOTS': NUM_ROBOTS,
        'POPULATION_SIZE': POPULATION_SIZE,
        'D': D
    })

    for run in inst.runs:
        GA(context, queue).execute(run)
