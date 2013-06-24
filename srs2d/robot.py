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

"""
This module implements a basic robot with differential wheels.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "20 Jun 2013"

import math
import random
import logging
import Box2D

__log__ = logging.getLogger(__name__)

class Robot(object):
    """
    Creates a robot with differential wheels.

    Usage:
        r = Robot(world, position=(4.1,2.0))
    """

    def __init__(self, world, position=None, angle=None):
        self._world = world
        self.initialPosition = position
        self.initialAngle = angle

        self.body = world.CreateDynamicBody()
        self.body.userData = self

        if position is not None:
            self.body.position = position

        if angle is not None:
            self.body.angle = angle

        self.fixture = self.body.CreateCircleFixture(radius=0.06, density=27)
        self.fixture.userData = self

        ## calculate a mask for the tires so they wont collide with the body
        #tiresMaskBits = (maskBits ^ (~categoryBits)) & 0x1111

        self.tires = ( Tire(self._world),
                       Tire(self._world) )

        self._world.CreateWeldJoint(bodyA=self.body, bodyB=self.tires[0].body,
                localAnchorA=Box2D.b2Vec2(-0.04425, 0), localAnchorB=Box2D.b2Vec2(0, 0))
        self._world.CreateWeldJoint(bodyA=self.body, bodyB=self.tires[1].body,
                localAnchorA=Box2D.b2Vec2(0.04425, 0), localAnchorB=Box2D.b2Vec2(0, 0))

    def reset(self, reset_position=True):
        """Reset the robot (stop)."""

        self.tires[0].reset()
        self.tires[1].reset()

        self.body.linearVelocity = Box2D.b2Vec2(0, 0)
        self.body.angularVelocity = 0

        if reset_position:
            self.set_position(self.initialPosition, angle=self.initialAngle)

    def set_position(self, position, angle=0.0):
        """Set the robot position and angle."""

        self.body.position = position
        self.body.angle = angle

        #self.tires[0].body.position = position - Box2D.b2Vec2(-0.04425, 0)
        #self.tires[1].body.position = position - Box2D.b2Vec2(0.04425, 0)

    def set_power(self, power):
        """Set the robot power for both motor. The 'power' parameter
        should be a (power0, power1) tuple where powerI is the power for the
        motor I. The power is a real number between -1.0 and 1.0."""

        self.tires[0].set_power(power[0])
        self.tires[1].set_power(power[1])

    def on_step(self):
        """Step callback."""

        self.tires[0].step()
        self.tires[1].step()

    def is_colliding(self):
        """Returns true if the robot is colliding with something else."""

        class callback(b2QueryCallback):
            def __init__(self, parent, **kwargs):
                b2QueryCallback.__init__(self, **kwargs)

                self.parent = parent
                self.hit = False

            def ReportFixture(self, fixture):
                if fixture == self.parent.fixture:
                    return True

                catA = fixture.filterData.categoryBits;
                maskA = fixture.filterData.maskBits;
                catB = self.parent.fixture.filterData.categoryBits;
                maskB = self.parent.fixture.filterData.maskBits;

                if ((catA & maskB) != 0 and (catB & maskA) != 0):
                    self.hit = True
                    return False

                return True

        cb = callback(self)
        self.world.QueryAABB(cb, self.fixture.GetAABB(0))

        return cb.hit

class Tire(object):
    """Implements a tire for the robot."""

    MAX_SPEED = 0.4
    DRIVE_FORCE = 0.2

    VERTICES = [ (-0.0085, 0.015), (0.0085, 0.015), 
                 (0.0085, -0.015), (-0.0085, -0.015) ]

    def __init__(self, world):
        self._world = world

        self.body = world.CreateDynamicBody()
        self.body.userData = self

        self.fixture = self.body.CreatePolygonFixture(
                vertices=self.VERTICES, density=1300*0.03)
        self.fixture.userData = self

        self.desired_speed = 0

    def reset(self):
        """Reset the tire (stop)."""

        self.desired_speed = 0
        self.body.linearVelocity = Box2D.b2Vec2(0, 0)
        self.body.angularVelocity = 0

    def on_step(self):
        """Step callback."""

        forward_normal = self.body.GetWorldVector(Box2D.b2Vec2(0,1))
        forward_speed = b2Dot(forward_normal, self.body.linearVelocity)
        forward_velocity = forward_speed * forward_normal

        lateral_normal = self.body.GetWorldVector(Box2D.b2Vec2(1,0))
        lateral_speed = b2Dot(lateral_normal, self.body.linearVelocity)
        lateral_velocity = lateral_speed * lateral_normal

        # apply necessary force
        force = 0
        if self.desired_speed > forward_speed:
            force = self.DRIVE_FORCE
        elif self.desired_speed < forward_speed:
            force = -self.DRIVE_FORCE

        if force != 0:
            self.body.ApplyForce(force * forward_normal, self.body.worldCenter, wake=True)

        # friction
        drag = 0.9
        drift = -4

        self.body.linearVelocity = (drag * forward_velocity) + (drift * lateral_velocity)

    def set_power(self, power):
        """Set the motor power for this tire. The power is a real number
        between -1.0 and 1.0."""

        speed = power * self.MAX_SPEED

        if (speed > self.MAX_SPEED):
            self.desired_speed = self.MAX_SPEED
        elif (speed < -self.MAX_SPEED):
            self.desired_speed = -self.MAX_SPEED

        self.desired_speed = speed
