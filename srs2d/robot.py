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

D2R = math.pi / 180.0

class Robot(physics.DynamicBody):
    def __init__(self, position=physics.Vector(0.0, 0.0)):
        super(Robot, self).__init__(position)

        self.add_shape(physics.CircleShape(radius=0.06, density=27))

        self.tires = DifferentialWheelsActuator(position=position, distance=0.0825, wheel_size=(0.017, 0.03))
        self.add(self.tires)

        self.front_led = LED(position=physics.Vector(0.0, 0.06), color=(255,0,0))
        self.add(self.front_led)

        self.rear_led = LED(position=physics.Vector(0.0, -0.06), color=(0,0,255))
        self.add(self.rear_led)

        self.camera = DualRegionCamera(origin=physics.Vector(0.0, 0.06), look_at=physics.Vector(0.0, 1.0))
        self.add(self.camera)

    @property
    def power(self):
        return (self.tires.power_left, self.tires.power_right)

    @power.setter
    def power(self, power):
        self.tires.power_left = power[0]
        self.tires.power_right = power[1]

class DifferentialWheelsActuator(physics.Actuator):
    MAX_SPEED = 0.4
    DRIVE_FORCE = 0.2
    DRAG = 0.9
    DRIFT = -4

    def __init__(self, position=physics.Vector(0.0, 0.0), distance=0.0845, wheel_size=(0.017, 0.03), wheel_density=1300):
        super(DifferentialWheelsActuator, self).__init__()

        self.position = position
        self.distance = distance

        x, y = wheel_size
        hx, hy = (x/2.0, y/2.0)
        wheel_vertices = [ (-hx, hy), (hx, hy),
                           (hx, -hy), (-hx, -hy) ]

        self.wheel_left = physics.DynamicBody(position=physics.Vector(position.x-(distance/2.0), position.y))
        self.wheel_right = physics.DynamicBody(position=physics.Vector(position.x+(distance/2.0), position.y))

        self.wheel_left.add_shape(physics.PolygonShape(vertices=wheel_vertices, density=wheel_density*0.03))
        self.wheel_right.add_shape(physics.PolygonShape(vertices=wheel_vertices, density=wheel_density*0.03))

        self.add(self.wheel_left)
        self.add(self.wheel_right)

        self._power_left = 0.0
        self._power_right = 0.0

    @property
    def power_left(self):
        return self._power_left

    @power_left.setter
    def power_left(self, value):
        if value > 1.0:
            self._power_left = 1.0
        elif value < -1.0:
            self._power_left = -1.0
        else:
            self._power_left = value

    @property
    def power_right(self):
        return self._power_right

    @power_right.setter
    def power_right(self, value):
        if value > 1.0:
            self._power_right = 1.0
        elif value < -1.0:
            self._power_right = -1.0
        else:
            self._power_right = value

    def on_added(self, parent):
        joint_left = physics.WeldJoint(target=parent, target_anchor=physics.Vector(-(self.distance/2.0), 0))
        self.wheel_left.add_joint(joint_left, anchor=physics.Vector(0.0, 0.0))

        joint_right = physics.WeldJoint(target=parent, target_anchor=physics.Vector((self.distance/2.0), 0))
        self.wheel_right.add_joint(joint_right, anchor=physics.Vector(0.0, 0.0))

    def on_step(self):
        self._step_wheel(self.wheel_left, self.power_left * self.MAX_SPEED)
        self._step_wheel(self.wheel_right, self.power_right * self.MAX_SPEED)

    def _step_wheel(self, wheel, desired_speed):
        """Calculate and apply forces on a tire."""
        forward_normal = wheel.world_vector(physics.Vector(0, 1))
        forward_speed = forward_normal.dot(wheel.linear_velocity)
        forward_velocity = forward_normal * forward_speed

        lateral_normal = wheel.world_vector(physics.Vector(1,0))
        lateral_speed = lateral_normal.dot(wheel.linear_velocity)
        lateral_velocity = lateral_normal * lateral_speed

        # apply necessary force
        force = 0
        if desired_speed > forward_speed:
            force = self.DRIVE_FORCE
        elif desired_speed < forward_speed:
            force = -self.DRIVE_FORCE

        if force != 0:
            wheel.apply_force(forward_normal * force, wheel.world_center, wake=True)

        wheel.linear_velocity = (forward_velocity * self.DRAG) + (lateral_velocity * self.DRIFT)

class LED(physics.ActuatorBody):
    def __init__(self, position=physics.Vector(0.0, 0.0), color=(255, 0, 0)):
        super(LED, self).__init__(position)

        self.color = color
        self._on = False

        self._rel_position = position

        self._shape = physics.CircleShape(radius=0.03, density=1, color=(0,0,0))
        self.add_shape(self._shape)

    def on_added(self, parent):
        joint = physics.WeldJoint(target=parent, target_anchor=self._rel_position)
        self.add_joint(joint, anchor=physics.Vector(0.0, 0.0))

    @property
    def on(self):
        return self._on

    @on.setter
    def on(self, value):
        self._on = value

        if self._on:
            self._shape.color = self.color
        else:
            self._shape.color = (0, 0, 0)

class DualRegionCamera(physics.RaycastSensor):
    def __init__(self, origin=physics.Vector(0.0, 0.0), look_at=physics.Vector(0.0, 0.0),
            fov=(72 * D2R), raycast_count=36, distance=0.65):
        super(DualRegionCamera, self).__init__(origin)

        self.look_at = look_at
        self.fov = fov
        self.raycast_count = raycast_count
        self.distance = distance

        adjacent = float(look_at.y - origin.y)
        opposite = float(look_at.x - origin.x)
        hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)

        angle_between = fov / float(raycast_count)
        angle_initial = math.asin(opposite / hypotenuse) - fov / 2.0 + angle_between / 2.0

        for i in range(raycast_count):
            angle = angle_initial + (angle_between * i)
            vertex = physics.Vector(distance * math.sin(angle), distance * math.cos(angle))
            self._vertices.append(vertex)

        self.values = {0: 0.0, 1: 0.0}

    def on_step(self):
        self.values[0] = 0.0
        self.values[1] = 0.0
        counter = 0

        for vertex in self._vertices:
            self.raycast_hit = False
            self.raycast_fraction = 1.0
            self.raycast_led = None
            self.raycast(self.body.transform * self._origin.to_b2Vec2(), self.body.transform * vertex.to_b2Vec2())

            if self.raycast_hit:
                region = 0 if counter < math.floor(len(self._vertices) / 2.0) else 1
                if self.raycast_led.color == (255, 0, 0):
                    self.values[region] += self.raycast_fraction
                else:
                    self.values[region] -= self.raycast_fraction

            counter += 1

        self.values[0] = self.values[0] / len(self._vertices)
        self.values[1] = self.values[1] / len(self._vertices)

    def callback(self, shape, point, normal, fraction):
        if shape.body is None:
            return fraction

        if isinstance(shape.body, LED):
            if shape.body.on:
                self.raycast_hit = True
                self.raycast_fraction = fraction
                self.raycast_led = shape.body

        return 0.0