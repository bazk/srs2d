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
__date__ = "04 Jul 2013"

import random
import logging
import physics
import pyopencl as cl
import logging.config
import logconfig
import solace

logging.config.dictConfig(logconfig.LOGGING)

__log__ = logging.getLogger(__name__)

NUM_SENSORS     = 13
NUM_ACTUATORS   = 4
NUM_HIDDEN      = 3

W = 0.9
ALFA = 2.0
BETA = 2.0

STEPS_TA = 18600
STEPS_TB = 5400

NUM_GENERATIONS = 1000
NUM_RUNS = 1
NUM_ROBOTS = 10
POPULATION_SIZE = 120

D = [0.7, 0.9, 1.1, 1.3, 1.5]

class PSO(object):
    def __init__(self):
        self.gbest = None
        self.gbest_fitness = None
        self.particles = []

    def execute(self, run, context, queue):
        __log__.info(' PSO Starting...')
        __log__.info('=' * 80)

        self.particles = [ Particle() for i in range(POPULATION_SIZE) ]
        self.simulator = physics.Simulator(context, queue, num_worlds=POPULATION_SIZE, num_robots=NUM_ROBOTS)

        generation = 0

        while (generation < NUM_GENERATIONS):
            __log__.info('Calculating fitness for each particle...')

            for p in range(len(self.particles)):
                pos = self.particles[p].position
                self.simulator.set_ann_parameters(p, pos)
            self.simulator.commit_ann_parameters()

            for p in range(len(self.particles)):
                self.particles[p].fitness = 0.0

            for d in D:
                for i in range(3):
                    self.simulator.init_worlds(d)
                    self.simulator.simulate(STEPS_TA, STEPS_TB)

                    fit = self.simulator.get_fitness()
                    for p in range(len(self.particles)):
                        self.particles[p].fitness += fit[p]

            for p in range(len(self.particles)):
                self.particles[p].fitness /= len(D) * 3

            for p in self.particles:
                p.update_pbest()

            for p in self.particles:
                if (self.gbest is None) or (p.pbest.fitness > self.gbest.fitness):
                    self.gbest = p.pbest.copy()

                    __log__.info('Found new gbest: %s', str(self.gbest))

            generation += 1

            __log__.info('-' * 80)
            __log__.info('[gen=%d] CURRENT GBEST IS: %s', generation, str(self.gbest))
            __log__.info(str(self.gbest.position))
            __log__.info('-' * 80)

            run.progress(generation / float(NUM_GENERATIONS), {'gbest_fitness': self.gbest.fitness, 'gbest_position': self.gbest.position.to_dict()})

            for p in self.particles:
                p.gbest = self.gbest
                p.update_pos_vel()

        run.done({'gbest_fitness': self.gbest.fitness, 'gbest_position': self.gbest.position.to_dict()})

class Particle(object):
    def __init__(self):
        self.id = id(self)

        self.position = physics.ANNParametersArray(True)
        self.velocity = physics.ANNParametersArray(True)
        self.fitness = 0.0

        self.pbest = None
        self.gbest = None

    def __str__(self):
        return 'Particle(%d, fitness=%.5f)' % (self.id, self.fitness)

    def copy(self):
        p = Particle()
        p.position = self.position.copy()
        p.velocity = self.velocity.copy()
        p.fitness = self.fitness
        p.pbest = self.pbest
        p.gbest = self.gbest
        return p

    def update_pbest(self):
        if (self.pbest is None) or (self.fitness > self.pbest.fitness):
            self.pbest = self.copy()

            __log__.info('[Particle %d] Found new pbest: %s', self.id, str(self.pbest))

    def update_pos_vel(self):
        global W, ALFA, BETA

        self.velocity = W * self.velocity + \
                        ALFA * random.uniform(0, 1.0) * (self.pbest.position - self.position) + \
                        BETA * random.uniform(0, 1.0) * (self.gbest.position - self.position)

        self.position = self.position + self.velocity

if __name__=="__main__":
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment('solace://lys:3000/2', 'admin', '123456')
    inst = exp.create_instance(NUM_RUNS, {
        'NUM_SENSORS': NUM_SENSORS,
        'NUM_ACTUATORS': NUM_ACTUATORS,
        'NUM_HIDDEN': NUM_HIDDEN,
        'W': W,
        'ALFA': ALFA,
        'BETA': BETA,
        'STEPS_TA': STEPS_TA,
        'STEPS_TB': STEPS_TB,
        'NUM_GENERATIONS': NUM_GENERATIONS,
        'NUM_RUNS': NUM_RUNS,
        'NUM_ROBOTS': NUM_ROBOTS,
        'POPULATION_SIZE': POPULATION_SIZE,
        'D': D
    })

    for run in inst.runs:
        PSO().execute(run, context, queue)
