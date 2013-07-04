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
import ann

__log__ = logging.getLogger(__name__)

D2R = math.pi / 180.0

class Robot(physics.DynamicBody):
    def __init__(self, position=physics.Vector(0.0, 0.0), **kwargs):
        super(Robot, self).__init__(position=position, **kwargs)

        self.add_shape(physics.CircleShape(radius=0.06, density=27))

        self.wheels = DifferentialWheelsActuator(id='wheels', position=position, distance=0.0825, wheel_size=(0.017, 0.03), mask_bits=0x0000)
        self.add(self.wheels)

        self.front_led = LED(id='front_led', position=physics.Vector(0.0, 0.06), color=(0,0,255), category_bits=0x0000, mask_bits=0x0000)
        self.add(self.front_led)

        self.rear_led = LED(id='rear_led', position=physics.Vector(0.0, -0.06), color=(255,0,0), category_bits=0x0000, mask_bits=0x0000)
        self.add(self.rear_led)

        self.camera = DualRegionCamera(id='camera', origin=physics.Vector(0.0, 0.06), look_at=physics.Vector(0.0, 1.0))
        self.add(self.camera)

        self.proximity = CircularProximitySensor(id='proximity', center=physics.Vector(0.0, 0.0), inner_radius=0.06, outer_radius=0.085, infrared_count=8)
        self.add(self.proximity)

        self.ground = BinaryGroundColorSensor(id='ground')
        self.add(self.ground)

        self.set_controller(ann.NeuralNetworkController())

class DifferentialWheelsActuator(physics.Actuator):
    MAX_SPEED = 0.4
    DRIVE_FORCE = 0.2
    DRAG = 0.9
    DRIFT = -4

    def __init__(self, position=physics.Vector(0.0, 0.0), distance=0.0845, wheel_size=(0.017, 0.03), wheel_density=1300, **kwargs):
        super(DifferentialWheelsActuator, self).__init__(**kwargs)

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

        self.connect('added', self.on_added)
        self.connect('step', self.on_step)

    def on_added(self, parent):
        joint_left = physics.WeldJoint(target=parent, target_anchor=physics.Vector(-(self.distance/2.0), 0))
        self.wheel_left.add_joint(joint_left, anchor=physics.Vector(0.0, 0.0))

        joint_right = physics.WeldJoint(target=parent, target_anchor=physics.Vector((self.distance/2.0), 0))
        self.wheel_right.add_joint(joint_right, anchor=physics.Vector(0.0, 0.0))

    def on_step(self):
        self._step_wheel(self.wheel_left, self._power_left * self.MAX_SPEED)
        self._step_wheel(self.wheel_right, self._power_right * self.MAX_SPEED)

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

    @property
    def values(self):
        return [self._power_left, self._power_right]

    @values.setter
    def values(self, values):
        def check_bounds(val):
            if val > 1.0:
                return 1.0
            elif val < -1.0:
                return -1.0
            else:
                return val

        self._power_left = check_bounds(values[0])
        self._power_right = check_bounds(values[1])

class LED(physics.Actuator):
    def __init__(self, position=physics.Vector(0.0, 0.0), color=(255, 0, 0), **kwargs):
        super(LED, self).__init__(**kwargs)

        self.color = color
        self._on = False
        self._rel_position = position

        self._body = physics.DynamicBody()
        self.add(self._body)

        self._shape = physics.CircleShape(radius=0.03, density=1, color=(0,0,0))
        self._body.add_shape(self._shape)

        self.connect('added', self.on_added)

    def on_added(self, parent):
        joint = physics.WeldJoint(target=parent, target_anchor=self._rel_position)
        self._body.add_joint(joint, anchor=physics.Vector(0.0, 0.0))

    @property
    def values(self):
        return [1.0] if self._on else [0.0]

    @values.setter
    def values(self, values):
        self._on = True if values[0] >= 0.5 else False

        if self._on:
            self._shape.color = self.color
        else:
            self._shape.color = (0, 0, 0)

class DualRegionCamera(physics.RaycastSensor):
    def __init__(self, origin=physics.Vector(0.0, 0.0), look_at=physics.Vector(0.0, 0.0),
            fov=(72 * D2R), raycast_count=36, distance=0.65, **kwargs):
        super(DualRegionCamera, self).__init__(**kwargs)

        self.origin = origin
        self.look_at = look_at
        self.fov = fov
        self.raycast_count = raycast_count
        self.distance = distance

        adjacent = float(look_at.y - origin.y)
        opposite = float(look_at.x - origin.x)
        hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)

        angle_between = fov / float(raycast_count)
        angle_initial = math.asin(opposite / hypotenuse) - fov / 2.0 + angle_between / 2.0

        self.vertices = []

        for i in range(raycast_count):
            angle = angle_initial + (angle_between * i)
            vertex = physics.Vector(distance * math.sin(angle), distance * math.cos(angle))
            self.vertices.append(vertex)

        self.connect('step', self.on_step)

        self.values = [ 0.0 for i in range(4) ]

    def on_step(self):
        new_values = [ 0.0 for i in range(4) ]
        counter = 0

        for vertex in self.vertices:
            self.raycast_hit = False
            self.raycast_fraction = 1.0
            self.raycast_led = None
            self.raycast(self.raycast_callback,
                self.parent.transform * self.origin.to_b2Vec2(),
                self.parent.transform * vertex.to_b2Vec2())

            if self.raycast_hit:
                region = 0 if counter < math.floor(len(self.vertices) / 2.0) else 2
                if self.raycast_led.color == (255, 0, 0):
                    new_values[region+0] = 1.0
                else:
                    new_values[region+1] = 1.0

            counter += 1

        def update_value(key, new_value):
            if self.values[key] != new_value:
                self.values[key] = new_value
                self.signal('update-value', key, new_value)

        for i in range(4):
            update_value(i, new_values[0])

    def raycast_callback(self, shape, point, normal, fraction):
        if shape.parent is None:
            return fraction

        body = shape.parent
        if isinstance(body, physics.DynamicBody):
            if body.parent is not None:
                if isinstance(body.parent, LED):
                    if body.parent._on:
                        self.raycast_hit = True
                        self.raycast_fraction = fraction
                        self.raycast_led = body.parent

        return 0.0

class CircularProximitySensor(physics.RaycastSensor):
    def __init__(self, center, inner_radius, outer_radius, infrared_count=8, **kwargs):
        super(CircularProximitySensor, self).__init__(**kwargs)

        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.infrared_count = infrared_count

        self.values = [ 0.0 for i in range(infrared_count) ]

        self.connect('step', self.on_step)

    def on_step(self):
        new_values = [ 0.0 for i in range(self.infrared_count) ]
        angle_between = 2 * math.pi / float(self.infrared_count)

        for i in range(self.infrared_count):
            self.raycast_hit = False
            self.raycast_fraction = 1.0

            angle = angle_between * i
            inner = physics.Vector(self.inner_radius * math.sin(angle), self.inner_radius * math.cos(angle))
            outer = physics.Vector(self.outer_radius * math.sin(angle), self.outer_radius * math.cos(angle))

            self.raycast(self.raycast_callback,
                self.parent.transform * inner.to_b2Vec2(),
                self.parent.transform * outer.to_b2Vec2())

            if self.raycast_hit:
                new_values[i] = self.raycast_fraction

        def update_value(key, new_value):
            if self.values[key] != new_value:
                self.values[key] = new_value
                self.signal('update-value', key, new_value)

        for i in range(self.infrared_count):
            update_value(i, new_values[i])

    def raycast_callback(self, shape, point, normal, fraction):
        if ((self.category_bits & shape.mask_bits) != 0) and ((shape.category_bits & self.mask_bits) != 0):
            self.raycast_hit = True
            self.raycast_fraction = fraction
            return 0.0

        return fraction

class ColorPadActuator(physics.Actuator):
    def __init__(self, center=physics.Vector(0.0, 0.0), radius=0.20, **kwargs):
        super(ColorPadActuator, self).__init__(**kwargs)

        self.center = center
        self.radius = radius

        self.connect('step', self.on_step)

    def on_step(self):
        self.world.signal('color-pad-notify', self)

class BinaryGroundColorSensor(physics.Sensor):
    def __init__(self, **kwargs):
        super(BinaryGroundColorSensor, self).__init__(**kwargs)

        self.values = [ 0.0 ]

        self.connect('realize', self.on_realize)
        self.connect('prepare', self.on_prepare)

    def on_realize(self, world):
        world.connect('color-pad-notify', self.on_color_pad_notify)

    def on_prepare(self):
        if self.values[0] != 0.0:
            self.values[0] = 0.0
            self.signal('update-value', 0, 0.0)

    def on_color_pad_notify(self, pad):
        x = self.parent.world_center.x
        y = self.parent.world_center.y

        if ((pad.center.x - x) ** 2 + (pad.center.y - y) ** 2) < (pad.radius ** 2):
            if self.values[0] != 1.0:
                self.values[0] = 1.0
                self.signal('update-value', 0, 1.0)