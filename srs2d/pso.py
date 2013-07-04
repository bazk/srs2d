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
import robot
import copy

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

class Simulation(object):
    def __init__(self):
        self.world = physics.World()

        D = [1.2, 1.5, 1.9, 2.3, 2.7]
        H = 4.20
        W = random.uniform(4.20, 4.90)
        d = D[int(math.floor(random.uniform(0, 4)))]
        x = math.sqrt(((d / 2.0) ** 2) / 2.0)
        self.world.add(robot.ColorPadActuator(center=physics.Vector(-x, x), radius=0.27))
        self.world.add(robot.ColorPadActuator(center=physics.Vector(x, -x), radius=0.27))

        VERTICAL_WALL_VERTICES = [ (-0.01, H/2.0), (0.01, H/2.0),
                                   (0.01, -H/2.0), (-0.01, -H/2.0) ]

        HORIZONTAL_WALL_VERTICES = [ (-W/2.0-0.01, 0.01), (W/2.0+0.01, 0.01),
                                     (W/2.0+0.01, -0.01), (-W/2.0-0.01, -0.01) ]

        wall = physics.StaticBody(position=physics.Vector(-W/2.0, 0))
        wall.add_shape(physics.PolygonShape(vertices=VERTICAL_WALL_VERTICES))
        self.world.add(wall)

        wall = physics.StaticBody(position=physics.Vector(0.0, -H/2.0))
        wall.add_shape(physics.PolygonShape(vertices=HORIZONTAL_WALL_VERTICES))
        self.world.add(wall)

        wall = physics.StaticBody(position=physics.Vector(W/2.0, 0))
        wall.add_shape(physics.PolygonShape(vertices=VERTICAL_WALL_VERTICES))
        self.world.add(wall)

        wall = physics.StaticBody(position=physics.Vector(0.0, H/2.0))
        wall.add_shape(physics.PolygonShape(vertices=HORIZONTAL_WALL_VERTICES))
        self.world.add(wall)

        NUM_ROBOTS = 12
        self.robots = [robot.Robot(position=physics.Vector( random.uniform(-W/2.0+0.12,W/2.0-0.12), random.uniform(-H/2.0+0.12, H/2.0-0.12) )) for i in range(NUM_ROBOTS)]
        for rob in self.robots:
            self.world.add(rob)

    def run(self, seconds):
        start = cur = self.world.clock
        while (cur - start) < seconds:
            self.world.step()
            cur = self.world.clock

class PSO(object):
    def __init__(self):
        self.gbest = None
        self.particles = []

class Particle(object):
    def __init__(self):
        self.pbest = None
        self.gbest = None

        self.position = PVector()
        self.velocity = PVector()

    def update(self):
        global W, ALFA, BETA

        self.velocity = W * self.velocity + \
                        ALFA * random.uniform(0, 1.0) * (self.pbest - self.position) + \
                        BETA * random.uniform(0, 1.0) * (self.gbest - self.position)

        self.position = self.position + self.velocity

class PVector(object):
    def __init__(self, duplicate=None, randomize=False, weights_boundary=(-5.0, 5.0), bias_boundary=(-5.0, 5.0), timec_boundary=(0, 1.0)):
        global INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS

        self.weights_boundary = weights_boundary
        self.bias_boundary = bias_boundary
        self.timec_boundary = timec_boundary

        if duplicate is not None:
            self.weights_boundary = copy.deepcopy(duplicate.weights_boundary)
            self.bias_boundary = copy.deepcopy(duplicate.bias_boundary)
            self.timec_boundary = copy.deepcopy(duplicate.timec_boundary)
            self.weights = copy.deepcopy(duplicate.weights)
            self.bias = copy.deepcopy(duplicate.bias)
            self.weights_hidden = copy.deepcopy(duplicate.weights_hidden)
            self.bias_hidden = copy.deepcopy(duplicate.bias_hidden)
            self.timec_hidden = copy.deepcopy(duplicate.timec_hidden)

        if randomize:
            self.weights = { o: { ih: random.uniform(weights_boundary[0], weights_boundary[1]) for ih in INPUT_NEURONS + HIDDEN_NEURONS } for o in OUTPUT_NEURONS }
            self.bias = { o: random.uniform(bias_boundary[0], bias_boundary[1]) for o in OUTPUT_NEURONS }
            self.weights_hidden = { h: { i: random.uniform(weights_boundary[0], weights_boundary[1]) for i in INPUT_NEURONS } for h in HIDDEN_NEURONS }
            self.bias_hidden = { h: random.uniform(bias_boundary[0], bias_boundary[1]) for h in HIDDEN_NEURONS }
            self.timec_hidden = { h: random.uniform(timec_boundary[0], timec_boundary[1]) for h in HIDDEN_NEURONS }

    def __radd__(self, other):
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

    def __rsub__(self, other):
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

    def __rmul__(self, other):
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

    def check_boundary(boundary, value):
        if value < boundary[0]:
            return boundary[0]
        elif value > boundary[1]:
            return boundary[1]
        else:
            return value

if __name__=="__main__":
    PSO().run()