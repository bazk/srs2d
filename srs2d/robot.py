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
import physics

__log__ = logging.getLogger(__name__)

class Robot(physics.DynamicBody):
    def __init__(self, position=(0.0, 0.0)):
        super(Robot, self).__init__(position)

        self.initialPosition = position
        self.initialAngle = angle

        self.add_shape(physics.CircleShape(radius=0.06, density=27))

        self.tires = DifferentialWheelsActuator(position=position, distance=0.04225, wheel_size=(0.03, 0.017))
        self.tires.attach(self.main_body)
        self.add(tires)

class DifferentialWheelsActuator(physics.Actuator):
    MAX_SPEED = 0.4
    DRIVE_FORCE = 0.2
    DRAG = 0.9
    DRIFT = -4

    def __init__(self, position=(0.0, 0.0), distance=0.04225, wheel_size=(0.03, 0.017), wheel_density=1300):
        super(Robot, self).__init__()

        self.position = position
        self.distance = distance

        x, y = wheel_size
        hx, hy = (x/2.0, y/2.0)
        wheel_vertices = [ (-hx, hy), (hx, hy),
                           (hx, -hy), (-hx, -hy) ]

        self.wheel_left = physics.DynamicBody(position=(position[0]-(distance/2.0), position[1]))
        self.wheel_right = physics.DynamicBody(position=(position[0]+(distance/2.0), position[1]))

        self.wheel_left.add_shape(physics.PolygonShape(vertices=wheel_vertices, density=wheel_density*0.03))
        self.wheel_right.add_shape(physics.PolygonShape(vertices=wheel_vertices, density=wheel_density*0.03))

        self.add(wheel_left)
        self.add(wheel_right)

        self.power_left = 0.0
        self.power_right = 0.0

    def on_add(self, parent):
        joint_left = physics.WeldJoint(target=parent, target_anchor=(-(self.distance/2.0), 0))
        self.wheel_left.add_joint(joint_left, anchor=(0.0, 0.0))

        joint_right = physics.WeldJoint(target=parent, target_anchor=((self.distance/2.0), 0))
        self.wheel_right.add_joint(joint_right, anchor=(0.0, 0.0))

    def on_step(self):
        self._step_wheel(self.wheel_left, self.power_left * self.MAX_SPEED)
        self._step_wheel(self.wheel_right, self.power_right * self.MAX_SPEED)

    def _step_wheel(self, wheel, desired_speed):
        """Calculate and apply forces on a tire."""

        forward_normal = tire.world_vector(0, 1)
        forward_speed = physics.dot(forward_normal, tire.linear_velocity)
        forward_velocity = forward_speed * forward_normal

        lateral_normal = tire.world_vector(1,0)
        lateral_speed = physics.dot(lateral_normal, tire.linear_velocity)
        lateral_velocity = lateral_speed * lateral_normal

        # apply necessary force
        force = 0
        if desired_speed > forward_speed:
            force = self.DRIVE_FORCE
        elif desired_speed < forward_speed:
            force = -self.DRIVE_FORCE

        if force != 0:
            tire.apply_force(force * forward_normal, tire.world_center, wake=True)

        tire.linear_velocity = (self.DRAG * forward_velocity) + (self.DRIFT * lateral_velocity)