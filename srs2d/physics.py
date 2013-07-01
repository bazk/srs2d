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

DEFAULT_CATEGORY_BITS = 0x1
DEFAULT_MASK_BITS = 0x1

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

        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.destructionListener = _DestructionListener(self)
        self.world.contactListener = _ContactListener(self)

        self.step_count = 0.0
        self.clock = 0.0

        self.children = []

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

    def add(self, child):
        child.added(self)
        child.realize(self.world)
        self.children.append(child)

    def step(self):
        """Run a single physics step."""
        self._lock.acquire()

        try:
            self.world.Step(self.time_step, self.velocity_iterations,
                    self.position_iterations)
            self.world.ClearForces()

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
    def __init__(self, x, y):
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

class Node(object):
    def __init__(self):
        self.children = []
        self.parent = None
        self.realized = False

    def add(self, child):
        self.children.append(child)
        child.added(self)

    def added(self, parent):
        self.parent = parent
        self.on_added(parent)

    def realize(self, b2World):
        self.on_realize(b2World)

        self.realized = True

        for child in self.children:
            child.realize(b2World)

    def step(self):
        self.on_step()

        for child in self.children:
            child.step()

    def on_added(self, parent):
        pass

    def on_realize(self, b2World):
        pass

    def on_step(self):
        pass


class DynamicBody(Node):
    def __init__(self, position=Vector(0.0, 0.0), categoryBits=None, maskBits=None):
        super(DynamicBody, self).__init__()
        self.position = position
        self.categoryBits = categoryBits
        self.maskBits = maskBits
        self.shapes = []
        self.joints = []

    def add_shape(self, shape):
        self.shapes.append(shape)
        shape.added(self)

    def add_joint(self, joint, anchor=Vector(0.0, 0.0)):
        self.joints.append(joint)
        joint.added(self, anchor)

    def on_added(self, parent):
        global DEFAULT_CATEGORY_BITS, DEFAULT_MASK_BITS

        super(DynamicBody, self).on_added(parent)

        if self.categoryBits is None:
            if isinstance(parent, DynamicBody):
                self.categoryBits = parent.categoryBits
            else:
                self.categoryBits = DEFAULT_CATEGORY_BITS

        if self.maskBits is None:
            if isinstance(parent, DynamicBody):
                self.maskBits = parent.maskBits
            else:
                self.maskBits = DEFAULT_MASK_BITS

    def on_realize(self, b2World):
        super(DynamicBody, self).on_realize(b2World)

        self._body = b2World.CreateDynamicBody(position=self.position.to_b2Vec2(),
            categoryBits=self.categoryBits, maskBits=self.maskBits)
        self._body.userData = self

        for shape in self.shapes:
            shape.on_realize(b2World)

        for joint in self.joints:
            joint.on_realize(b2World)

    def to_b2Body(self):
        return self._body

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

        return Vector(center.x, center.y)

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
    def __init__(self):
        super(Actuator, self).__init__()


class Sensor(Node):
    def __init__(self):
        super(Sensor, self).__init__()

class RaycastSensor(Sensor):
    def __init__(self, origin=Vector(0.0, 0.0)):
        super(RaycastSensor, self).__init__()

        self._origin = origin
        self._vertices = []

    def on_realize(self, b2World):
        super(RaycastSensor, self).on_realize(b2World)
        self._world = b2World

    def raycast(self, origin, dest):
        callback = self._RaycastCallback(self)
        self._world.RayCast(callback, origin, dest)

    def callback(self, shape, point, normal, fraction):
        pass

    class _RaycastCallback(Box2D.b2RayCastCallback):
        def __init__(self, parent, **kwargs):
            Box2D.b2RayCastCallback.__init__(self)
            self.parent = parent

        def ReportFixture(self, fixture, point, normal, fraction):
            return self.parent.callback(fixture.userData, point, normal, fraction)

    @property
    def vertices(self):
        transformed = []

        for vertex in self._vertices:
            transformed.append(self.parent.transform * vertex.to_b2Vec2())

        return transformed

    @property
    def origin(self):
        return self.parent.transform * self._origin.to_b2Vec2()

class Shape(Node):
    def __init__(self, density=1, color=(255, 0, 0)):
        super(Shape, self).__init__()
        self.density = density
        self.color = color

class PolygonShape(Shape):
    def __init__(self, vertices=[], density=1, color=(64, 196, 64)):
        super(PolygonShape, self).__init__(density, color)
        self._vertices = vertices

    def on_realize(self, b2World):
        super(PolygonShape, self).on_realize(b2World)
        self._fixture = self.parent.to_b2Body().CreatePolygonFixture(
            vertices=self._vertices, density=self.density)
        self._fixture.userData = self

    @property
    def vertices(self):
        transformed = []

        for vertex in self._vertices:
            transformed.append(self.parent.transform * vertex)

        return transformed

class CircleShape(Shape):
    def __init__(self, radius=1, density=1, color=(64, 196, 64)):
        super(CircleShape, self).__init__(density, color)
        self.radius = radius

    def on_realize(self, b2World):
        super(CircleShape, self).on_realize(b2World)
        self._fixture = self.parent.to_b2Body().CreateCircleFixture(
            radius=self.radius, density=self.density)
        self._fixture.userData = self

    @property
    def center(self):
        return self.parent.transform * self._fixture.shape.pos

    @property
    def orientation(self):
        return self.parent.transform.R.col2


class Joint(Node):
    def added(self, parent, anchor):
        self.parent = parent
        self.on_added(parent, anchor)

    def on_added(self, parent, anchor):
        pass

class WeldJoint(Joint):
    def __init__(self, target=None, target_anchor=None):
        super(WeldJoint, self).__init__()
        if target is None or target_anchor is None:
            return

        self.body1 = target
        self.body1_anchor = target_anchor

    def on_added(self, parent, anchor):
        super(WeldJoint, self).on_added(parent, anchor)
        self.body2 = parent
        self.body2_anchor = anchor

    def on_realize(self, b2World):
        super(WeldJoint, self).on_realize(b2World)
        b2World.CreateWeldJoint(bodyA=self.body1.to_b2Body(), localAnchorA=self.body1_anchor.to_b2Vec2(),
                bodyB=self.body2.to_b2Body(), localAnchorB=self.body2_anchor.to_b2Vec2())
