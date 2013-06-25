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

class Simulator(object):
    """
    Creates a 2D top-down physics Simulator.

    Usage:

        sim = Simulator()

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

        self.objects = []

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

    def step(self):
        """Run a single physics step."""
        self._lock.acquire()

        try:
            self.world.Step(self.time_step, self.velocity_iterations,
                    self.position_iterations)
            self.world.ClearForces()

            for obj in self.objects:
                obj.on_step()

            self.on_step()

            for obj in self.objects:
                obj.update_shapes()

            self.step_count += 1
            self.clock += self.time_step

        finally:
            self._lock.release()

    def register_object(self, obj):
        self.objects.append(obj)
        self.current_id_seq += 1
        return self.current_id_seq - 1

    def on_step(self):
        """Called after every single step."""
        pass

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

class DynamicObject(object):
    def __init__(self, simulator):
        self.simulator = simulator
        self.id = simulator.register_object(self)
        self.color = (254, 0, 0)
        self.bodies = []
        self.shapes = []

    def add(self, body):
        self.bodies.append(body)

    def on_step(self):
        pass

    def update_shapes(self):
        self.shapes = []

        for body in self.bodies:
            transform = body.transform

            for fixture in body.fixtures:
                b2shape = fixture.shape
                shape = None

                if isinstance(b2shape, Box2D.b2PolygonShape):
                    shape = PolygonShape()

                    for vertex in b2shape.vertices:
                        transformed = transform * vertex
                        shape.vertices.append((transformed.x, transformed.y))

                elif isinstance(b2shape, Box2D.b2CircleShape):
                    shape = CircleShape()

                    center = Box2D.b2Mul(transform, b2shape.pos)
                    orientation = transform.R.col2

                    shape.center = (center.x, center.y)
                    shape.radius = b2shape.radius
                    shape.orientation = (orientation.x, orientation.y)

                else:
                    shape = Shape()

                shape.object_id = self.id
                shape.color = self.color

                self.shapes.append(shape)

class Shape(object):
    object_id = None
    color = (234, 0, 0)

class PolygonShape(Shape):
    def __init__(self):
        super(PolygonShape, self).__init__()
        self.vertices = []

class CircleShape(Shape):
    def __init__(self):
        super(CircleShape, self).__init__()
        self.center = (0.0, 0.0)
        self.radius = 1
        self.orientation = None