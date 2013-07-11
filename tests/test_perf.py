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
__date__ = "19 Jun 2013"

import unittest
import random
import time
import srs2d.physics
import srs2d.robot

class BasePerfTest(unittest.TestCase):
    def setUp(self):
        self.world = srs2d.physics.World()

        self.H = H = 4.20
        self.W = W = 4.90

        self.world.add(srs2d.robot.ColorPadActuator(center=srs2d.physics.Vector(-0.7, 0.7), radius=0.27))
        self.world.add(srs2d.robot.ColorPadActuator(center=srs2d.physics.Vector(0.7, -0.7), radius=0.27))

        VERTICAL_WALL_VERTICES = [ srs2d.physics.Vector(-0.01, H/2.0),
                                   srs2d.physics.Vector(0.01, H/2.0),
                                   srs2d.physics.Vector(0.01, -H/2.0),
                                   srs2d.physics.Vector(-0.01, -H/2.0) ]

        HORIZONTAL_WALL_VERTICES = [ srs2d.physics.Vector(-W/2.0-0.01, 0.01),
                                     srs2d.physics.Vector(W/2.0+0.01, 0.01),
                                     srs2d.physics.Vector(W/2.0+0.01, -0.01),
                                     srs2d.physics.Vector(-W/2.0-0.01, -0.01) ]

        wall = srs2d.physics.StaticBody(position=srs2d.physics.Vector(-W/2.0, 0))
        wall.add_shape(srs2d.physics.PolygonShape(vertices=VERTICAL_WALL_VERTICES))
        self.world.add(wall)

        wall = srs2d.physics.StaticBody(position=srs2d.physics.Vector(0.0, -H/2.0))
        wall.add_shape(srs2d.physics.PolygonShape(vertices=HORIZONTAL_WALL_VERTICES))
        self.world.add(wall)

        wall = srs2d.physics.StaticBody(position=srs2d.physics.Vector(W/2.0, 0))
        wall.add_shape(srs2d.physics.PolygonShape(vertices=VERTICAL_WALL_VERTICES))
        self.world.add(wall)

        wall = srs2d.physics.StaticBody(position=srs2d.physics.Vector(0.0, H/2.0))
        wall.add_shape(srs2d.physics.PolygonShape(vertices=HORIZONTAL_WALL_VERTICES))
        self.world.add(wall)

    def run_simulation(self, seconds):
        start = cur = self.world.clock
        while (cur - start) < seconds:
            self.world.step()
            cur = self.world.clock

class LowRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 5
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')

class MidRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 10
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')

class MidHighRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 15
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')

class HighRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 20
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')


class UltraRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 25
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')


class UltraHighRobotCountPerfTest(BasePerfTest):
    def runTest(self):
        NUM_ROBOTS = 30
        SIM_TIME = 10

        start = time.time()

        self.robots = [ srs2d.robot.Robot(position=srs2d.physics.Vector(
            random.uniform(-self.W/2.0+0.12, self.W/2.0-0.12),
            random.uniform(-self.H/2.0+0.12, self.H/2.0-0.12) )) for i in range(NUM_ROBOTS) ]

        for rob in self.robots:
            self.world.add(rob)

        self.run_simulation(SIM_TIME)

        print 'Simulation took ', time.time() - start, ' seconds'

        self.assertLess(time.time() - start, SIM_TIME, 'simulation slower than real time')