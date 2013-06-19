# -*- coding: utf-8 -*-
#
# This file is part of trooper-simulator.
#
# trooper-simulator is free software: you can redistribute it and/or modify
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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "19 Jun 2013"

import unittest
import srs2d.physics

class SimpleWorldTestCase(unittest.TestCase):
    def setUp(self):
        self.world = srs2d.physics.World()

class InitialWorldTestCase(SimpleWorldTestCase):
    def runTest(self):
        (step_count, clock, shapes) = self.world.get_state()
        self.assertEqual(step_count, 0, 'incorrect initial step_count')
        self.assertEqual(clock, 0.0, 'incorrect initial clock')
        self.assertEqual(shapes, [], 'initial shapes list has objects')

class OneStepWorldTestCase(SimpleWorldTestCase):
    def runTest(self):
        self.world.step()
        (step_count, clock, shapes) = self.world.get_state()
        self.assertEqual(step_count, 1, 'incorrect step_count after one step')
        self.assertEqual(clock, self.world.time_step,
                'incorrect clock after one step')
        self.assertEqual(shapes, [], 'shapes list has objects after one step')
