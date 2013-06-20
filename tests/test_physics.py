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
import srs2d.physics

class ExtendedSimulator(srs2d.physics.Simulator):
    """Extend the simulator to mark which callbacks were called."""

    called_on_step = False
    called_on_pre_solve = False
    called_on_begin_contact = False
    called_on_end_contact = False
    called_on_post_solve = False
    called_on_destroy = False

    def on_step(self):
        self.called_on_step = True

    def on_pre_solve(self, contact, old_manifold):
        self.called_on_pre_solve = True

    def on_begin_contact(self, contact):
        self.called_on_begin_contact = True

    def on_end_contact(self, contact):
        self.called_on_end_contact = True

    def on_post_solve(self, contact, impulse):
        self.called_on_post_solve = True

    def on_destroy(self, obj):
        self.called_on_destroy = True

class EmptySimulator(unittest.TestCase):
    """Setup an empty simulator."""

    def setUp(self):
        self.simulator = ExtendedSimulator()

class EmptyTestCase(EmptySimulator):
    """Check if step() is working."""

    def runTest(self):
        (step_count, clock, shapes) = self.simulator.get_state()
        self.assertEqual(step_count, 0, 'incorrect initial step_count')
        self.assertEqual(clock, 0.0, 'incorrect initial clock')
        self.assertEqual(shapes, [], 'initial shapes list has objects')

        self.simulator.step()
        
        (step_count, clock, shapes) = self.simulator.get_state()
        self.assertEqual(step_count, 1, 'incorrect step_count after one step')
        self.assertEqual(clock, self.simulator.time_step,
                'incorrect clock after one step')
        self.assertEqual(shapes, [], 'shapes list has objects after one step')

class OneBodySimulator(EmptySimulator):
    """Extend the empty simulatori setup adding one body."""

    def setUp(self):
        super(OneBodySimulator, self).setUp()
        self.body1 = self.simulator.world.CreateDynamicBody()
        self.fixture1 = self.body1.CreateCircleFixture(radius=0.06, density=27)

class OneBodyTestCase(OneBodySimulator):
    """Checks if on_step was called."""

    def runTest(self):
        self.assertEqual(self.simulator.called_on_step,
                False, 'on_step called before time')

        self.simulator.step()

        self.assertEqual(self.simulator.called_on_step,
                True, 'on_step not called')

        (step_count, clock, shapes) = self.simulator.get_state()
        self.assertEqual(step_count, 1, 'incorrect step_count after one step')
        self.assertEqual(clock, self.simulator.time_step,
                'incorrect clock after one step')
        self.assertNotEqual(shapes, [], 'shapes list does not have any objects')

class TwoBodiesSimulator(unittest.TestCase):
    """Setup a simulator with two bodies not colliding"""

    def setUp(self):
        self.simulator = ExtendedSimulator()

        self.body1 = self.simulator.world.CreateDynamicBody(position=(0,0))
        self.fixture1 = self.body1.CreateCircleFixture(radius=0.06, density=27)

        self.body2 = self.simulator.world.CreateDynamicBody(position=(1,0))
        self.fixture2 = self.body2.CreateCircleFixture(radius=0.06, density=27)

class TwoBodiesTestCase(TwoBodiesSimulator):
    """Checks if no calls to collision methods happened (since the bodies
    are not colliding)."""

    def runTest(self):
        self.assertEqual(self.simulator.called_on_step,
                False, 'on_step called before time')

        self.simulator.step()

        self.assertEqual(self.simulator.called_on_step,
                True, 'on_step not called')
        self.assertEqual(self.simulator.called_on_pre_solve,
                False, 'on_pre_solve called')
        self.assertEqual(self.simulator.called_on_begin_contact,
                False, 'on_begin_contact called')
        self.assertEqual(self.simulator.called_on_end_contact,
                False, 'on_end_contact called')
        self.assertEqual(self.simulator.called_on_post_solve,
                False, 'on_post_solve called')
        self.assertEqual(self.simulator.called_on_destroy,
                False, 'on_destroy called')

class ThreeBodiesCollisionTestCase(TwoBodiesSimulator):
    """Add one more body colliding with the other ones and check if collision
    methods get called."""

    def setUp(self):
        super(ThreeBodiesCollisionTestCase, self).setUp()

        self.body3 = self.simulator.world.CreateDynamicBody(position=(0.05,0))
        self.fixture3 = self.body3.CreateCircleFixture(radius=0.06, density=27)

    def runTest(self):
        self.assertEqual(self.simulator.called_on_step,
                False, 'on_step called before time')

        self.simulator.step()

        self.assertEqual(self.simulator.called_on_step,
                True, 'on_step not called')
        self.assertEqual(self.simulator.called_on_pre_solve,
                True, 'on_pre_solve not called')
        self.assertEqual(self.simulator.called_on_begin_contact,
                True, 'on_begin_contact not called')
        self.assertEqual(self.simulator.called_on_end_contact,
                False, 'on_end_contact called')
        self.assertEqual(self.simulator.called_on_post_solve,
                True, 'on_post_solve not called')
        self.assertEqual(self.simulator.called_on_destroy,
                False, 'on_destroy called')
