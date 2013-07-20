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

import sys
import math
import random
import logging
import physics
import copy
import time
import multiprocessing
import pyopencl as cl
import numpy as np

__log__ = logging.getLogger(__name__)

NUM_SENSORS     = 13
NUM_ACTUATORS   = 4
NUM_HIDDEN      = 3

W = 0.9
ALFA = 2
BETA = 2

SIMULATION_DURATION = 600
NUM_ROBOTS = 30
D = [1.2, 1.5, 1.9, 2.3, 2.7]

class PSO(object):
    def __init__(self):
        self.gbest = None
        self.gbest_fitness = None
        self.particles = []

    def run(self, population_size=42, max_generations=1000):
        print 'PSO Starting...'
        print '==============='

        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        self.particles = [ Particle() for i in range(population_size) ]
        self.simulator = physics.Simulator(context, queue, num_worlds=population_size, num_robots=NUM_ROBOTS)

        generation = 0

        while (generation < max_generations):
            print 'Calculating fitness for each particle...'

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

            print 'Updating pbest for each particle...'
            for p in self.particles:
                p.update_pbest()

            print 'Updating gbest...'
            for p in self.particles:
                if (self.gbest is None) or (p.pbest.fitness > self.gbest.fitness):
                    self.gbest = p.pbest.copy()

                    print 'Found new gbest: ', str(self.gbest)

            print '-' * 80
            print 'CURRENT GBEST IS: ', str(self.gbest)
            print str(self.gbest.position)
            print '-' * 80

            print 'Calculating new position and velocity for each particle...'
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

            print '[Particle %d] Found new pbest: %s' % (self.id, str(self.pbest))

    def update_pos_vel(self):
        global W, ALFA, BETA

        self.velocity = W * self.velocity + \
                        ALFA * random.uniform(0, 1.0) * (self.pbest.position - self.position) + \
                        BETA * random.uniform(0, 1.0) * (self.gbest.position - self.position)

        self.position = self.position + self.velocity

if __name__=="__main__":
    PSO().run()