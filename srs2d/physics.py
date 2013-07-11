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
This module provides a simple physics wrapper for PyBox2D.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "19 Jun 2013"

import threading
import logging
import Box2D

__log__ = logging.getLogger(__name__)

DEFAULT_CATEGORY_BITS = 0x0001
DEFAULT_MASK_BITS = 0xFFFF

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

class Node(object):
    def __init__(self, id=None, category_bits=None, mask_bits=None):
        self.id = id
        self.children = []
        self.parent = None
        self.realized = False

        self.category_bits = category_bits
        self.mask_bits = mask_bits

        self.registered_callbacks = {}

    def signal(self, signal_name, *args, **kwargs):
        if not signal_name in self.registered_callbacks:
            return False

        for cb in self.registered_callbacks[signal_name]:
            cb(*args, **kwargs)

    def connect(self, signal_name, callback_function):
        if not signal_name in self.registered_callbacks:
            self.registered_callbacks[signal_name] = [callback_function]
        else:
            self.registered_callbacks[signal_name].append(callback_function)

    def add(self, child):
        self.children.append(child)
        child.added(self)
        self.signal('add' , child)

    def added(self, parent):
        self.parent = parent
        self.signal('added' , parent)

    def realize(self, world):
        global DEFAULT_CATEGORY_BITS, DEFAULT_MASK_BITS

        if self.category_bits is None:
            if isinstance(self.parent, Node):
                self.category_bits = self.parent.category_bits
            else:
                self.category_bits = DEFAULT_CATEGORY_BITS

        if self.mask_bits is None:
            if isinstance(self.parent, Node):
                self.mask_bits = self.parent.mask_bits
            else:
                self.mask_bits = DEFAULT_MASK_BITS

        self.world = world
        self.signal('realize', world)

        for child in self.children:
            child.realize(world)

        self.realized = True

    def prepare(self):
        self.signal('prepare')

        for child in self.children:
            child.prepare()

    def step(self):
        self.signal('step')

        for child in self.children:
            child.step()

    def think(self):
        self.signal('think')

        for child in self.children:
            child.think()

    def bounding_rectangle(self):
        rect = None

        for child in self.children:
            child_rect = child.bounding_rectangle()

            if child_rect is None:
                continue

            if rect is None:
                rect = child_rect
            else:
                rect.union(child_rect)

        return rect

    def filter(self, cls):
        ret = []

        if isinstance(self, cls):
            ret.append(self)

        for child in self.children:
            ret.extend(child.filter(cls))

        return ret

class World(Node):
    """
    Creates a 2D top-down physics World.

    Usage:

        sim = World()

        while 1:
            sim.step()
            (step_count, clock, shapes) = sim.get_state()
            # draw_the_screen(shapes)
    """

    def __init__(self):
        super(World, self).__init__()
        self.time_step = 1.0 / 30.0
        self.velocity_iterations = 3
        self.position_iterations = 1
        self.step_count = 0.0
        self.clock = 0.0

        self._b2World = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.category_bits = DEFAULT_CATEGORY_BITS
        self.mask_bits = DEFAULT_MASK_BITS
        self.realized = True

        self.connect('add', self._on_add)

    def to_b2World(self):
        return self._b2World

    def step(self):
        """Run a single physics step."""

        self.signal('prepare')
        for child in self.children:
            child.prepare()

        self._b2World.Step(self.time_step, self.velocity_iterations,
                self.position_iterations)
        self._b2World.ClearForces()

        self.signal('step')
        for child in self.children:
            child.step()

        self.step_count += 1
        self.clock += self.time_step

        self.signal('think')
        for child in self.children:
            child.think()

    def _on_add(self, child):
        child.realize(self)


class Controller(Node):
    def __init__(self):
        super(Controller, self).__init__()
        self._sensor_nodes = []
        self._actuator_nodes = []
        self.sensors = {}
        self.actuators = {}

        self.connect('added', self._on_added)

    def _on_added(self, parent):
        self._sensor_nodes = parent.filter(Sensor)
        self._actuator_nodes = parent.filter(Actuator)

    def update_sensors(self):
        for sensor in self._sensor_nodes:
            if sensor.id is None:
                continue

            values = sensor.values
            for i in range(len(values)):
                self.sensors[sensor.id + str(i)] = values[i]

    def update_actuators(self):
        for actuator in self._actuator_nodes:
            if actuator.id is None:
                continue

            values = actuator.values
            new_values = [ 0.0 for i in range(len(values)) ]
            for i in range(len(values)):
                new_values[i] = self.actuators[actuator.id + str(i)]

            actuator.values = new_values

class Vector(object):
    def __init__(self, x=None, y=None):
        if x is None:
            raise ValueError("first positional argument can not be 'None'")

        if isinstance(x, Box2D.b2Vec2):
            self.x = x.x
            self.y = x.y
        elif isinstance(x, tuple):
            self.x, self.y = x
        else:
            if y is None:
                raise ValueError("parameter 'y' can not be 'None'")

            self.x = x
            self.y = y

    def to_b2Vec2(self):
        return Box2D.b2Vec2((self.x, self.y))

    def dot(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y

        raise ArithmeticError('cannot dot product vector by non-vector')

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, long) or isinstance(other, float):
            return Vector(other * self.x, other * self.y)

        raise ArithmeticError('cannot multiply vector by non-numeric')

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)

        raise ArithmeticError('cannot dot product vector by non-vector')

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

class BoundingRectangle(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def union(self, other):
        if other is not None:
            if other.low.x < self.low.x: self.low.x = other.low.x
            if other.low.y < self.low.y: self.low.y = other.low.y
            if other.high.x > self.high.x: self.high.x = other.high.x
            if other.high.y > self.high.y: self.high.y = other.high.y

class AttrDef(object):
    def __init__(self, obj, key, read_only=True):
        self.obj = obj
        self.key = key
        self.read_only = read_only
        self.getter = getattr(obj, 'get_'+key)
        if not self.read_only:
            self.setter = getattr(obj, 'set_'+key)
        else:
            self.setter = None

class Body(Node):
    def __init__(self, position=Vector(0.0, 0.0), **kwargs):
        super(Body, self).__init__(**kwargs)
        self.position = position
        self.shapes = []
        self.joints = []
        self._body = None
        self._world_center = Vector(0, 0)

    def add_shape(self, shape):
        self.shapes.append(shape)
        shape.added(self)

    def add_joint(self, joint, anchor=Vector(0.0, 0.0)):
        self.joints.append(joint)
        joint.added(self, anchor)

    def to_b2Body(self):
        return self._body

    def bounding_rectangle(self):
        rect = None

        for shape in self.shapes:
            shape_rect = shape.bounding_rectangle()

            if shape_rect is None:
                continue

            if rect is None:
                rect = shape_rect
            else:
                rect.union(shape_rect)

        for child in self.children:
            child_rect = child.bounding_rectangle()

            if child_rect is None:
                continue

            if rect is None:
                rect = child_rect
            else:
                rect.union(child_rect)

        return rect

    def world_vector(self, local_vector):
        if self._body is None:
            return local_vector

        world_vector = self._body.GetWorldVector(local_vector.to_b2Vec2())

        return Vector(world_vector.x, world_vector.y)

    @property
    def world_center(self):
        if self._body is None:
            return Vector(0.0, 0.0)

        center = self._body.worldCenter
        self._world_center.x = center.x
        self._world_center.y = center.y

        return self._world_center

    @property
    def transform(self):
        if self._body is None:
            return None

        return self._body.transform

class DynamicBody(Body):
    def __init__(self, **kwargs):
        super(DynamicBody, self).__init__(**kwargs)
        self.controller = None

        self.connect('realize', self._on_realize)
        self.connect('think', self._on_think)

    def set_controller(self, controller):
        self.controller = controller
        controller.added(self)

    def _on_realize(self, world):
        self._body = world.to_b2World().CreateDynamicBody(position=self.position.to_b2Vec2())
        self._body.userData = self

        for shape in self.shapes:
            shape.realize(world)

        for joint in self.joints:
            joint.realize(world)

    def _on_think(self):
        if self.controller is not None:
            self.controller.think()

    @property
    def linear_velocity(self):
        if self._body is None:
            return Vector(0.0, 0.0)

        vel = self._body.linearVelocity

        return Vector(vel.x, vel.y)

    @linear_velocity.setter
    def linear_velocity(self, vector):
        if self._body is None:
            return

        self._body.linearVelocity = vector.to_b2Vec2()

    def apply_force(self, force, position, wake=True):
        if self._body is None:
            return

        return self._body.ApplyForce(force.to_b2Vec2(), position.to_b2Vec2(), wake=wake)

class StaticBody(Body):
    def __init__(self, **kwargs):
        super(StaticBody, self).__init__(**kwargs)
        self.connect('realize', self._on_realize)

    def _on_realize(self, world):
        self._body = world.to_b2World().CreateStaticBody(position=self.position.to_b2Vec2())
        self._body.userData = self

        for shape in self.shapes:
            shape.realize(world)

        for joint in self.joints:
            joint.realize(world)


class Actuator(Node):
    def __init__(self, **kwargs):
        super(Actuator, self).__init__(**kwargs)


class Sensor(Node):
    def __init__(self, **kwargs):
        super(Sensor, self).__init__(**kwargs)


class RaycastSensor(Sensor):
    def raycast(self, callback, origin, dest):
        cb_instance = self._RaycastCallback(callback)
        self.world.to_b2World().RayCast(cb_instance, origin, dest)

    class _RaycastCallback(Box2D.b2RayCastCallback):
        def __init__(self, callback, **kwargs):
            Box2D.b2RayCastCallback.__init__(self)
            self.callback = callback

        def ReportFixture(self, fixture, point, normal, fraction):
            return self.callback(fixture.userData, Vector(point), Vector(normal), fraction)

class Shape(Node):
    def __init__(self, density=1, color=(64, 255, 0), **kwargs):
        super(Shape, self).__init__(**kwargs)
        self.density = density
        self.color = color

class PolygonShape(Shape):
    def __init__(self, vertices=[], **kwargs):
        super(PolygonShape, self).__init__(**kwargs)
        self._vertices = vertices

        self.connect('realize', self._on_realize)

    def _on_realize(self, world):
        vert = [ v.to_b2Vec2() for v in self._vertices ]
        self._fixture = self.parent.to_b2Body().CreatePolygonFixture(
            vertices=vert, density=self.density,
            categoryBits=self.category_bits, maskBits=self.mask_bits)
        self._fixture.userData = self

    @property
    def vertices(self):
        transformed = []

        for vertex in self._vertices:
            transformed.append(self.parent.transform * vertex.to_b2Vec2())

        return transformed

    def bounding_rectangle(self):
        vertices = self.vertices
        low = Vector(0, 0)
        high = Vector(0, 0)
        low.x, low.y = (vertices[0].x, vertices[0].y)
        high.x, high.y = (vertices[0].x, vertices[0].y)

        for vertex in vertices[1:]:
            if vertex.x < low.x:  low.x = vertex.x
            if vertex.y < low.y:  low.y = vertex.y
            if vertex.x > high.x: high.x = vertex.x
            if vertex.y > high.y: high.y = vertex.y

        return BoundingRectangle(low, high)

class CircleShape(Shape):
    def __init__(self, radius=1, **kwargs):
        super(CircleShape, self).__init__(**kwargs)
        self.radius = radius

        self.connect('realize', self._on_realize)

    def _on_realize(self, world):
        self._fixture = self.parent.to_b2Body().CreateCircleFixture(
            radius=self.radius, density=self.density,
            categoryBits=self.category_bits, maskBits=self.mask_bits)
        self._fixture.userData = self

    @property
    def center(self):
        return self.parent.transform * self._fixture.shape.pos

    @property
    def orientation(self):
        return self.parent.transform.R.col2

    def bounding_rectangle(self):
        center = self.center
        low = Vector(center.x - self.radius, center.y - self.radius)
        high = Vector(center.x + self.radius, center.y + self.radius)
        return BoundingRectangle(low, high)

class Joint(Node):
    def added(self, parent, anchor):
        self.parent = parent
        self.signal('added', parent, anchor)

class WeldJoint(Joint):
    def __init__(self, target=None, target_anchor=None, **kwargs):
        super(WeldJoint, self).__init__(**kwargs)

        if target is None or target_anchor is None:
            raise ValueError("'target' and 'target_anchor' can not be 'None'")

        self.body1 = target
        self.body1_anchor = target_anchor

        self.connect('added', self._on_added)
        self.connect('realize', self._on_realize)

    def _on_added(self, parent, anchor):
        self.body2 = parent
        self.body2_anchor = anchor

    def _on_realize(self, world):
        world.to_b2World().CreateWeldJoint(bodyA=self.body1.to_b2Body(), localAnchorA=self.body1_anchor.to_b2Vec2(),
                bodyB=self.body2.to_b2Body(), localAnchorB=self.body2_anchor.to_b2Vec2())
