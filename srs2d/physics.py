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

class World(object):
    """
    Creates a 2D top-down physics World.

    Usage:

        sim = World()

        while 1:
            sim.step()
            (step_count, clock, shapes) = sim.get_state()
            # draw_the_screen(shapes)
    """

    time_step = 1.0 / 30.0
    velocity_iterations = 3
    position_iterations = 1

    current_id_seq = 0

    def __init__(self):
        __log__.info("Initializing physics...")

        self._lock = threading.Lock()

        self._b2World = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self._b2World.destructionListener = _DestructionListener(self)
        self._b2World.contactListener = _ContactListener(self)

        self.step_count = 0.0
        self.clock = 0.0

        self.children = []

        self.registered_callbacks = {}

        __log__.info("Initialization complete.")

    def reset(self):
        """Reset counters and clock (does not reset the world itself)."""

        self._lock.acquire()

        try:
            __log__.info("Reset.")
            self.step_count = 0.0
            self.clock = 0.0
        finally:
            self._lock.release()

    def to_b2World(self):
        return self._b2World

    def add(self, child):
        child.added(self)
        child.realize(self)
        self.children.append(child)

    def signal(self, signal_name, parameter=None):
        if not signal_name in self.registered_callbacks:
            return False

        for cb in self.registered_callbacks[signal_name]:
            if parameter is None:
                cb()
            else:
                cb(parameter)

    def register(self, signal_name, callback_function):
        if not signal_name in self.registered_callbacks:
            self.registered_callbacks[signal_name] = [callback_function]
        else:
            self.registered_callbacks[signal_name].append(callback_function)

    def step(self):
        """Run a single physics step."""
        self._lock.acquire()

        try:
            self._b2World.Step(self.time_step, self.velocity_iterations,
                    self.position_iterations)
            self._b2World.ClearForces()

            for node in self.children:
                node.step()

            self.step_count += 1
            self.clock += self.time_step

        finally:
            self._lock.release()

    def on_pre_solve(self, contact, old_manifold):
        """Called before a contact gets resolved."""
        pass

    def on_begin_contact(self, contact):
        """Called on the beginning of a contact."""
        pass

    def on_end_contact(self, contact):
        """Called when ending a contact"""
        pass

    def on_post_solve(self, contact, impulse):
        """Called after a contact is resolved."""
        pass

    def on_destroy(self, obj):
        """Called when an object is destroyed."""
        pass

class _DestructionListener(Box2D.b2DestructionListener):
    """Wrapper for b2DestructionListener."""
    def __init__(self, parent, **kwargs):
        super(_DestructionListener, self).__init__(**kwargs)
        self.parent = parent

    def SayGoodbye(self, obj):
        self.parent.on_destroy(obj)

class _ContactListener(Box2D.b2ContactListener):
    """Wrapper for b2ContactListener."""
    def __init__(self, parent, **kwargs):
        super(_ContactListener, self).__init__(**kwargs)
        self.parent = parent

    def PreSolve(self, contact, old_manifold):
        self.parent.on_pre_solve(contact, old_manifold)

    def BeginContact(self, contact):
        self.parent.on_begin_contact(contact)

    def EndContact(self, contact):
        self.parent.on_end_contact(contact)

    def PostSolve(self, contact, impulse):
        self.parent.on_post_solve(contact, impulse)


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

class Node(object):
    def __init__(self, category_bits=None, mask_bits=None):
        self.children = []
        self.parent = None
        self.realized = False

        self.category_bits = category_bits
        self.mask_bits = mask_bits

    def add(self, child):
        self.children.append(child)
        child.added(self)

    def added(self, parent):
        self.parent = parent
        self.on_added(parent)

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
        self.on_realize(world)
        self.realized = True

        for child in self.children:
            child.realize(world)

    def step(self):
        self.on_step()

        for child in self.children:
            child.step()

    def on_added(self, parent):
        pass

    def on_realize(self, world):
        pass

    def on_step(self):
        pass

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

class Object(Node):
    def __init__(self, **kwargs):
        super(Object, self).__init__(**kwargs)

        self.attributes = []

    def add_attribute(self, obj, key, **kwargs):
        attr_def = AttrDef(obj, key, **kwargs)
        self.attributes.append(attr_def)

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

class DynamicBody(Node):
    def __init__(self, position=Vector(0.0, 0.0), **kwargs):
        super(DynamicBody, self).__init__(**kwargs)
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

    def on_realize(self, world):
        super(DynamicBody, self).on_realize(world)

        self._body = world.to_b2World().CreateDynamicBody(position=self.position.to_b2Vec2())
        self._body.userData = self

        for shape in self.shapes:
            shape.realize(world)

        for joint in self.joints:
            joint.realize(world)

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

    def apply_force(self, force, position, wake=True):
        if self._body is None:
            return

        return self._body.ApplyForce(force.to_b2Vec2(), position.to_b2Vec2(), wake=wake)

    @property
    def transform(self):
        if self._body is None:
            return None

        return self._body.transform

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

    def on_realize(self, world):
        super(PolygonShape, self).on_realize(world)
        self._fixture = self.parent.to_b2Body().CreatePolygonFixture(
            vertices=self._vertices, density=self.density,
            categoryBits=self.category_bits, maskBits=self.mask_bits)
        self._fixture.userData = self

    @property
    def vertices(self):
        transformed = []

        for vertex in self._vertices:
            transformed.append(self.parent.transform * vertex)

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

    def on_realize(self, world):
        super(CircleShape, self).on_realize(world)
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
        self.on_added(parent, anchor)

    def on_added(self, parent, anchor):
        pass

class WeldJoint(Joint):
    def __init__(self, target=None, target_anchor=None, **kwargs):
        super(WeldJoint, self).__init__(**kwargs)

        if target is None or target_anchor is None:
            raise ValueError("'target' and 'target_anchor' can not be 'None'")

        self.body1 = target
        self.body1_anchor = target_anchor

    def on_added(self, parent, anchor):
        super(WeldJoint, self).on_added(parent, anchor)
        self.body2 = parent
        self.body2_anchor = anchor

    def on_realize(self, world):
        super(WeldJoint, self).on_realize(world)
        world.to_b2World().CreateWeldJoint(bodyA=self.body1.to_b2Body(), localAnchorA=self.body1_anchor.to_b2Vec2(),
                bodyB=self.body2.to_b2Body(), localAnchorB=self.body2_anchor.to_b2Vec2())
