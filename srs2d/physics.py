# -*- coding: utf-8 -*-
#
# This file is part of trooper-simulator.
#
# trooper-simulator is free software: you can redistribute it and/or modify
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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides a simple physics wrapper for PyBox2D.

Check BasicPhysics class documentation for more information.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "19 Jun 2013"

import threading
import logging
import Box2D

__log__ = logging.getLogger(__name__)

class BasicPhysics(object):
    """
    BasicPhysics is just a PyBox2D wrapper used as basis for the simulator.

    Usage:

        p = BasicPhysics()

        while 1:
            p.step()
            (step_count, clock, shapes) = p.get_state()
            # draw_the_screen(shapes)
    """

    time_step = 1.0 / 30.0
    velocity_iterations = 3
    position_iterations = 1

    def __init__(self):
        __log__.info("Initializing physics...")

        self.lock = threading.Lock()

        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.destructionListener = _DestructionListener(self)
        self.world.contactListener = _ContactListener(self)

        self.step_count = 0.0
        self.clock = 0.0

        __log__.info("Initialization complete.")

    def reset(self):
        """Reset counters and clock (does not reset the world itself)."""

        self.lock.acquire()

        try:
            __log__.info("Reset.")
            self.step_count = 0.0
            self.clock = 0.0
        finally:
            self.lock.release()

    def step(self):
        """Run a single physics step."""
        self.lock.acquire()

        try:
            self.world.Step(self.time_step, self.velocity_iterations,
                    self.position_iterations)
            self.on_step()
            self.step_count += 1
            self.clock += self.time_step

        finally:
            self.lock.release()

    def simulate(self, seconds):
        """Simply calls self.step() until reaches the wanted number of seconds
        (in simulation time, not real time)."""

        __log__.info("Simulating %d seconds...", seconds)

        target = self.step_count + (seconds / self.time_step)

        while self.step_count < target:
            self.step()

        __log__.info("Simulation complete.")

    def get_state(self):
        """Returns the state of the physics world as
        (step_count, clock, shapes) tuple."""

        state = None

        self.lock.acquire()

        try:
            shapes = []

            for body in self.world.bodies:
                self.__get_body_shapes(body, shapes)

            state = (self.step_count, self.clock, shapes)
        finally:
            self.lock.release()

        return state

    @staticmethod
    def __get_body_shapes(body, shapes):
        """Extends 'shapes' with 'body' shapes."""

        transform = body.transform

        for fixture in body.fixtures:
            trackid = None
            color = None

            if fixture.userData is not None:
                if hasattr(fixture.userData, 'drawShapes'):
                    shapes.extend(fixture.userData.drawShapes)

                if hasattr(fixture.userData, 'id'):
                    trackid = fixture.userData.id

                if hasattr(fixture.userData, 'color'):
                    color = fixture.userData.color

            shape = fixture.shape

            if color is None and body.userData is not None \
                    and hasattr(fixture.userData, 'color'):
                color = body.userData.color

            shape_def = None

            if isinstance(shape, Box2D.b2PolygonShape):
                vertices = []
                for vertex in shape.vertices:
                    transformed = transform * vertex
                    vertices.append((transformed.x, transformed.y))

                shape_def = { 'type': 'polygon',
                             'group': 'bodies',
                             'color': color,
                             'vertices': vertices }

            elif isinstance(shape, Box2D.b2CircleShape):
                center = Box2D.b2Mul(transform, shape.pos)
                orientation = transform.R.col2

                shape_def = {
                        'type': 'circle',
                        'group': 'bodies',
                        'color': color,
                        'center': (center.x, center.y),
                        'radius': shape.radius,
                        'orientation': (orientation.x, orientation.y) }

            if shape_def is not None:
                if trackid is not None:
                    shape_def['track'] = trackid

                shapes.append(shape_def)

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
        self.parent.on_egin_contact(contact)

    def EndContact(self, contact):
        self.parent.on_end_contact(contact)

    def PostSolve(self, contact, impulse):
        self.parent.on_post_solve(contact, impulse)
