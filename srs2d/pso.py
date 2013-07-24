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

logging.config.dictConfig(logconfig.LOGGING)

__log__ = logging.getLogger(__name__)

NUM_SENSORS     = 13
NUM_ACTUATORS   = 4
NUM_HIDDEN      = 3

W = 0.9
ALFA = 2.0
BETA = 2.0

SIMULATION_DURATION = 600
NUM_ROBOTS = 30
D = [2.2, 2.5, 2.9, 3.3, 3.7]

class PSO(object):
    def __init__(self):
        self.gbest = None
        self.gbest_fitness = None
        self.particles = []

    def run(self, population_size=40, max_generations=2000):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        __log__.info(' PSO Starting...')
        __log__.info('=' * 80)
        __log__.info(' population_size = %d', population_size)
        __log__.info(' max_generations = %d', max_generations)
        __log__.info(' NUM_ROBOTS = %d', NUM_ROBOTS)
        __log__.info(' SIMULATION_DURATION = %d', SIMULATION_DURATION)
        __log__.info(' W = %f', W)
        __log__.info(' ALFA = %f', ALFA)
        __log__.info(' BETA = %f', BETA)
        __log__.info('=' * 80)

        self.particles = [ Particle() for i in range(population_size) ]
        self.simulator = physics.Simulator(context, queue, num_worlds=population_size, num_robots=NUM_ROBOTS)

        generation = 0

        while (generation < max_generations):
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
                    self.simulator.simulate(SIMULATION_DURATION)

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

            __log__.info('-' * 80)
            __log__.info('[gen=%d] CURRENT GBEST IS: %s', generation, str(self.gbest))
            __log__.info(str(self.gbest.position))
            __log__.info('-' * 80)

            for p in self.particles:
                p.gbest = self.gbest
                p.update_pos_vel()

            generation += 1

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
    PSO().run()
