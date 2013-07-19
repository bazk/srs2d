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

    def run(self, population_size=8, max_generations=3):
        print 'PSO Starting...'
        print '==============='

        context = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(context)

        self.particles = [ Particle() for i in range(population_size) ]
        self.worlds = physics.World(context, queue, num_worlds=population_size, num_robots=NUM_ROBOTS)

        generation = 0

        while (generation < max_generations):
            print 'Calculating fitness for each particle...'
            for p in range(len(self.particles)):
                pos = self.particles[p].position
                self.worlds.set_ann_parameters(p, pos.weights, pos.bias, pos.weights_hidden,
                    pos.bias_hidden, pos.timec_hidden)

            self.worlds.commit_ann_parameters()
            self.worlds.simulate(SIMULATION_DURATION)

            fit = self.worlds.get_fitness()
            for p in range(len(self.particles)):
                self.particles[p].fitness = fit[p]

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

        self.position = PVector(True)
        self.velocity = PVector(True)
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

class PVector(object):
    def __init__(self, randomize=False, weights_boundary=(-5.0, 5.0), bias_boundary=(-5.0, 5.0), timec_boundary=(0, 1.0)):
        self.weights_boundary = weights_boundary
        self.bias_boundary = bias_boundary
        self.timec_boundary = timec_boundary

        if randomize:
            self.weights = np.random.uniform(weights_boundary[0], weights_boundary[1], NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN))
            self.bias = np.random.uniform(bias_boundary[0], bias_boundary[1], NUM_ACTUATORS)
            self.weights_hidden = np.random.uniform(weights_boundary[0], weights_boundary[1], NUM_HIDDEN * NUM_SENSORS)
            self.bias_hidden = np.random.uniform(bias_boundary[0], bias_boundary[1], NUM_HIDDEN)
            self.timec_hidden = np.random.uniform(timec_boundary[0], timec_boundary[1], NUM_HIDDEN)

        else:
            self.weights = np.zeros(NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN), dtype=np.float32)
            self.bias = np.zeros(NUM_ACTUATORS, dtype=np.float32)
            self.weights_hidden = np.zeros(NUM_HIDDEN * NUM_SENSORS, dtype=np.float32)
            self.bias_hidden = np.zeros(NUM_HIDDEN, dtype=np.float32)
            self.timec_hidden = np.zeros(NUM_HIDDEN, dtype=np.float32)

    def __str__(self):
        return str({
            'weights': [ x for x in self.weights.flat ],
            'bias': [ x for x in self.bias.flat ],
            'weights_hidden': [ x for x in self.weights_hidden.flat ],
            'bias_hidden': [ x for x in self.bias_hidden.flat ],
            'timec_hidden': [ x for x in self.timec_hidden.flat ]
        })

    def copy(self):
        pv = PVector()
        pv.weights_boundary = copy.deepcopy(self.weights_boundary)
        pv.bias_boundary = copy.deepcopy(self.bias_boundary)
        pv.timec_boundary = copy.deepcopy(self.timec_boundary)
        pv.weights = np.copy(self.weights)
        pv.bias = np.copy(self.bias)
        pv.weights_hidden = np.copy(self.weights_hidden)
        pv.bias_hidden = np.copy(self.bias_hidden)
        pv.timec_hidden = np.copy(self.timec_hidden)
        return pv

    def __add__(self, other):
        if isinstance(other, PVector):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights + other.weights
            ret.bias = self.bias + other.bias
            ret.weights_hidden = self.weights_hidden + other.weights_hidden
            ret.bias_hidden = self.bias_hidden + other.bias_hidden
            ret.timec_hidden = self.timec_hidden + other.timec_hidden

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __sub__(self, other):
        if isinstance(other, PVector):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights - other.weights
            ret.bias = self.bias - other.bias
            ret.weights_hidden = self.weights_hidden - other.weights_hidden
            ret.bias_hidden = self.bias_hidden - other.bias_hidden
            ret.timec_hidden = self.timec_hidden - other.timec_hidden

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights * other
            ret.bias = self.bias * other
            ret.weights_hidden = self.weights_hidden * other
            ret.bias_hidden = self.bias_hidden * other
            ret.timec_hidden = self.timec_hidden * other

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            return self.__mul__(other)
        else:
            raise NotImplemented

    @staticmethod
    def check_boundary(boundary, array):
        for i in range(array.size):
            if array[i] < boundary[0]:
                array[i] = boundary[0]
            elif array[i] > boundary[1]:
                array[i] = boundary[1]

    def export(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'weights_hidden': self.weights_hidden,
            'bias_hidden': self.bias_hidden,
            'timec_hidden': self.timec_hidden
        }

if __name__=="__main__":
    PSO().run()