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
import physics

__log__ = logging.getLogger(__name__)

class Robot(physics.DynamicObject):
    """
    Creates a robot with differential wheels.

    Usage:
        r = Robot(world, position=(4.1,2.0))
    """

    MAX_SPEED = 0.4
    DRIVE_FORCE = 0.2

    # friction
    DRAG = 0.9
    DRIFT = -4

    TIRE_VERTICES = [ (-0.0085, 0.015), (0.0085, 0.015),
                      (0.0085, -0.015), (-0.0085, -0.015) ]

    def __init__(self, simulator, position=(0, 0), angle=0.0):
        super(Robot, self).__init__(simulator)

        self.initialPosition = position
        self.initialAngle = angle

        self.body = simulator.world.CreateDynamicBody(position=position)
        self.body.userData = self

        self.fixture = self.body.CreateCircleFixture(radius=0.06, density=27)
        self.fixture.userData = self

        self.tires = ( simulator.world.CreateDynamicBody(position=(position[0]-0.04225, position[1])),
                       simulator.world.CreateDynamicBody(position=(position[0]+0.04225, position[1])) )

        self.desired_speed = {0: 0.0, 1: 0.0}

        for tire in self.tires:
            fixture = tire.CreatePolygonFixture(vertices=self.TIRE_VERTICES, density=1300*0.03)
            tire.userData = self
            fixture.userData = self

        simulator.world.CreateWeldJoint(bodyA=self.body, bodyB=self.tires[0],
                localAnchorA=Box2D.b2Vec2(-0.04425, 0), localAnchorB=Box2D.b2Vec2(0, 0))
        simulator.world.CreateWeldJoint(bodyA=self.body, bodyB=self.tires[1],
                localAnchorA=Box2D.b2Vec2(0.04425, 0), localAnchorB=Box2D.b2Vec2(0, 0))

        self.add(self.body)
        for tire in self.tires:
            self.add(tire)

        self.update_shapes()

    def reset(self, reset_position=True):
        """Reset the robot (stop)."""

        for tire in self.tires:
            self.desired_speed[0] = 0.0
            self.desired_speed[1] = 0.0
            tire.linearVelocity = Box2D.b2Vec2(0, 0)
            tire.angularVelocity = 0

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

    def set_motor_power(self, motor, power):
        """Set the motor power for one motor. The 'motor' parameter is an
        integer 0 or 1, representing left and right motor, respectively. The
        'power' is a real number between -1.0 and 1.0."""

        speed = power * self.MAX_SPEED

        if (speed > self.MAX_SPEED):
            self.desired_speed[motor] = self.MAX_SPEED
        elif (speed < -self.MAX_SPEED):
            self.desired_speed[motor] = -self.MAX_SPEED
        else:
            self.desired_speed[motor] = speed

    def on_step(self):
        """Step callback."""

        self._step_tire(self.tires[0], self.desired_speed[0])
        self._step_tire(self.tires[1], self.desired_speed[1])

    def _step_tire(self, tire, desired_speed):
        """Calculate and apply forces on a tire."""

        forward_normal = tire.GetWorldVector(Box2D.b2Vec2(0,1))
        forward_speed = Box2D.b2Dot(forward_normal, tire.linearVelocity)
        forward_velocity = forward_speed * forward_normal

        lateral_normal = tire.GetWorldVector(Box2D.b2Vec2(1,0))
        lateral_speed = Box2D.b2Dot(lateral_normal, tire.linearVelocity)
        lateral_velocity = lateral_speed * lateral_normal

        # apply necessary force
        force = 0
        if desired_speed > forward_speed:
            force = self.DRIVE_FORCE
        elif desired_speed < forward_speed:
            force = -self.DRIVE_FORCE

        if force != 0:
            tire.ApplyForce(force * forward_normal, tire.worldCenter, wake=True)

        tire.linearVelocity = (self.DRAG * forward_velocity) + (self.DRIFT * lateral_velocity)

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
