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

__log__ = logging.getLogger(__name__)

INPUT_NEURONS = [
    'camera0',
    'camera1',
    'camera2',
    'camera3',
    'proximity3',
    'proximity2',
    'proximity1',
    'proximity0',
    'proximity7',
    'proximity6',
    'proximity5',
    'proximity4',
    'ground0'
]

HIDDEN_NEURONS = [
    'hidden0',
    'hidden1',
    'hidden2'
]

OUTPUT_NEURONS = [
    'wheels1',
    'wheels0',
    'rear_led0',
    'front_led0'
]

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

    def run(self, population_size=8):
        print 'PSO Starting...'
        print '==============='

        context = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(context)

        self.particles = [ Particle() for i in range(population_size) ]
        self.worlds = physics.World(context, queue, num_worlds=population_size, num_robots=NUM_ROBOTS)

        while True:
            print 'Calculating fitness for each particle...'
            # for p in self.particles:
            #     p.socket.send(p.position.export())

            self.worlds.simulate(SIMULATION_DURATION)
            time.sleep(1) # just so the computer dont freeze

            for p in self.particles:
                p.fitness = random.uniform(0,10)

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
        global INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS

        self.weights_boundary = weights_boundary
        self.bias_boundary = bias_boundary
        self.timec_boundary = timec_boundary

        if randomize:
            self.weights = { o: { ih: random.uniform(weights_boundary[0], weights_boundary[1]) for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            self.bias = { o: random.uniform(bias_boundary[0], bias_boundary[1]) for o in OUTPUT_NEURONS }
            self.weights_hidden = { h: { i: random.uniform(weights_boundary[0], weights_boundary[1]) for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            self.bias_hidden = { h: random.uniform(bias_boundary[0], bias_boundary[1]) for h in HIDDEN_NEURONS }
            self.timec_hidden = { h: random.uniform(timec_boundary[0], timec_boundary[1]) for h in HIDDEN_NEURONS }

        else:
            self.weights = { o: { ih: 0.0 for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            self.bias = { o: 0.0 for o in OUTPUT_NEURONS }
            self.weights_hidden = { h: { i: 0.0 for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            self.bias_hidden = { h: 0.0 for h in HIDDEN_NEURONS }
            self.timec_hidden = { h: 0.0 for h in HIDDEN_NEURONS }

    def __str__(self):
        return str({
            'weights': self.weights,
            'bias': self.bias,
            'weights_hidden': self.weights_hidden,
            'bias_hidden': self.bias_hidden,
            'timec_hidden': self.timec_hidden
        })

    def copy(self):
        pv = PVector()
        pv.weights_boundary = copy.deepcopy(self.weights_boundary)
        pv.bias_boundary = copy.deepcopy(self.bias_boundary)
        pv.timec_boundary = copy.deepcopy(self.timec_boundary)
        pv.weights = copy.deepcopy(self.weights)
        pv.bias = copy.deepcopy(self.bias)
        pv.weights_hidden = copy.deepcopy(self.weights_hidden)
        pv.bias_hidden = copy.deepcopy(self.bias_hidden)
        pv.timec_hidden = copy.deepcopy(self.timec_hidden)
        return pv

    def __add__(self, other):
        if isinstance(other, PVector):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary
            ret.weights = { o: { ih: self.check_boundary(self.weights_boundary, self.weights[o][ih]+other.weights[o][ih]) for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            ret.bias = { o: self.check_boundary(self.bias_boundary, self.bias[o]+other.bias[o]) for o in OUTPUT_NEURONS }
            ret.weights_hidden = { h: { i: self.check_boundary(self.weights_boundary, self.weights_hidden[h][i]+other.weights_hidden[h][i]) for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            ret.bias_hidden = { h: self.check_boundary(self.bias_boundary, self.bias_hidden[h]+other.bias_hidden[h]) for h in HIDDEN_NEURONS }
            ret.timec_hidden = { h: self.check_boundary(self.timec_boundary, self.timec_hidden[h]+other.timec_hidden[h]) for h in HIDDEN_NEURONS }
            return ret
        else:
            raise NotImplemented

    def __sub__(self, other):
        if isinstance(other, PVector):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary
            ret.weights = { o: { ih: self.check_boundary(self.weights_boundary, self.weights[o][ih]-other.weights[o][ih]) for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            ret.bias = { o: self.check_boundary(self.bias_boundary, self.bias[o]-other.bias[o]) for o in OUTPUT_NEURONS }
            ret.weights_hidden = { h: { i: self.check_boundary(self.weights_boundary, self.weights_hidden[h][i]-other.weights_hidden[h][i]) for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            ret.bias_hidden = { h: self.check_boundary(self.bias_boundary, self.bias_hidden[h]-other.bias_hidden[h]) for h in HIDDEN_NEURONS }
            ret.timec_hidden = { h: self.check_boundary(self.timec_boundary, self.timec_hidden[h]-other.timec_hidden[h]) for h in HIDDEN_NEURONS }
            return ret
        else:
            raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            ret = PVector()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary
            ret.weights = { o: { ih: self.check_boundary(self.weights_boundary, self.weights[o][ih]*other) for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            ret.bias = { o: self.check_boundary(self.bias_boundary, self.bias[o]*other) for o in OUTPUT_NEURONS }
            ret.weights_hidden = { h: { i: self.check_boundary(self.weights_boundary, self.weights_hidden[h][i]*other) for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            ret.bias_hidden = { h: self.check_boundary(self.bias_boundary, self.bias_hidden[h]*other) for h in HIDDEN_NEURONS }
            ret.timec_hidden = { h: self.check_boundary(self.timec_boundary, self.timec_hidden[h]*other) for h in HIDDEN_NEURONS }
            return ret
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            return self.__mul__(other)
        else:
            raise NotImplemented

    @staticmethod
    def check_boundary(boundary, value):
        if value < boundary[0]:
            return boundary[0]
        elif value > boundary[1]:
            return boundary[1]
        else:
            return value

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