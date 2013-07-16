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
Provides a viewer for the physics module.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "21 Jun 2013"

import os
import sys
import time
import math
import random
import logging
import threading
import pygame
import Box2D
import pyopencl as cl

import dropdown
from .. import physics

__log__ = logging.getLogger(__name__)

class Scene(object):
    do_exit = False
    exit_callback = None

    world = None
    running = False
    do_step = False

    target_fps = 30.0

    selected_obj = None
    selection_change_callback = None

    def __init__(self, background=(0,0,0), resolution=(800,600)):
        self.background = background
        self.resolution = resolution

        pygame.init()

        try:
            self.font = pygame.font.Font(None, 15)
        except IOError:
            try:
                self.font = pygame.font.Font('freesansbold.ttf', 15)
            except IOError:
                log.warn("Unable to load default font or 'freesansbold.ttf', disabling text.")
                self.write = lambda *args: 0

        pygame.display.set_caption('SRS2d Viewer')

        self.screen = pygame.display.set_mode(self.resolution)
        self.surface = pygame.Surface(self.resolution, pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

        self.dropdown = dropdown.DropDown(self.font)
        self.dropdown.add_item('Add robot', callback=self.add_robot)

        self.zoom = 120
        self.center = Box2D.b2Vec2(0, 0)
        self.offset = (-self.resolution[0]/2, -self.resolution[1]/2)

        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        self.world = physics.World(context, queue)
        self.transforms = self.world.get_transforms()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def step(self):
        self.do_step = True

    def is_running(self):
        return self.running

    def add_robot(self, pos):
        # rob = robot.Robot(position=self._to_world(pos))
        # self.world.add(rob)
        # # rob.power = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        # # rob.front_led.on = random.uniform(0,1) > 0.5
        # # rob.rear_led.on = random.uniform(0,1) > 0.5
        # self.robots.append(rob)
        pass

    def on_mouse_down(self, event):
        self.dropdown.on_mouse_down(event)

    def on_mouse_up(self, event):
        if not self.dropdown.on_mouse_up(event):
            if event.button == 1:
                new_obj = self._check_selection(event)

                if new_obj != self.selected_obj:
                    if self.selection_change_callback is not None:
                        self.selection_change_callback(new_obj)

                self.selected_obj = new_obj

                return (self.selected_obj is not None)

    def on_mouse_move(self, event):
        self.dropdown.on_mouse_move(event)

    def _check_selection(self, event):
        # for obj in self.world.children: # just nodes attached directly in world
        #     if not isinstance(obj, physics.Body):
        #         continue

        #     square = obj.bounding_rectangle()

        #     # convert to screen points (swap high_y and low_y because of the screen y-flip)
        #     low_x, high_y = self._to_screen(square.low)
        #     high_x, low_y = self._to_screen(square.high)

        #     if (event.x >= low_x) and (event.y >= low_y) and \
        #        (event.x <= high_x) and (event.y <= high_y):
        #         return obj

        return None

    def on_select_shape(self, shape):
        print 'on_select_shape'

    def on_deselect_shape(self, shape):
        print 'on_deselect_shape'

    def draw(self):
        if self.running or self.do_step:
            self.world.step(1.0 / self.target_fps)
            self.do_step = False

            self.transforms = self.world.get_transforms()

        self.screen.fill(self.background)
        self.surface.fill(self.background)
        self.__write_pos = 30

        if self.transforms is not None:
            for transform in self.transforms:
                center = self._to_screen((transform[0], transform[1]))
                radius = 0.06 * self.zoom
                orientation = (transform[2], transform[3])
                self.draw_circle(center, radius, orientation=orientation, fill=(255,0,0,128), border=(255,0,0,255))

        self.dropdown.draw(self.surface)

        real_clock = self.world.clock

        self.write(str(self.clock.get_fps()), (200,80,80))
        self.write("%02d:%02d:%02d" % (int(real_clock) / 3600,
                                      (int(real_clock) % 3600) / 60,
                                       int(real_clock) % 60), (80,80,200))

        self.screen.blit(self.surface, (0, 0))
        self.clock.tick(self.target_fps)
        pygame.display.flip()

    def _to_screen(self, point):
        """Transform a world point to screen point."""

        x = ((point[0] + self.center.x) * self.zoom) - self.offset[0]
        y = ((point[1] + self.center.y) * self.zoom) - self.offset[1]
        y = self.resolution[1] - y

        return (int(x), int(y))

    def _to_world(self, point):
        """Transform a screen point to world point."""

        (x, y) = point
        y = self.resolution[1] - y
        x = (float(x + self.offset[0]) / self.zoom) - self.center.x
        y = (float(y + self.offset[1]) / self.zoom) - self.center.y

        return (x, y)

    def draw_robot(self, robot):
        center = self._to_screen(robot.transform.pos)
        radius = robot.body_radius * self.zoom
        orientation = (robot.transform.rot.sin, robot.transform.rot.cos)
        self.draw_circle(center, radius, orientation=orientation, fill=(255,0,0,128), border=(255,0,0,255))

    def draw_nodes_recursive(self, node):
        if node.realized:
            self.draw_node(node)

            for node in node.children:
                self.draw_nodes_recursive(node)

    def draw_node(self, node):
        if node == self.selected_obj:
            square = node.bounding_rectangle()
            vertices = [
                self._to_screen(square.low),
                self._to_screen(physics.Vector(square.high.x, square.low.y)),
                self._to_screen(square.high),
                self._to_screen(physics.Vector(square.low.x, square.high.y))
            ]
            self.draw_polygon(vertices, border=(64, 0, 255, 255))

        if isinstance(node, physics.Body):
            for shape in node.shapes:
                self.draw_shape(shape)

        # elif isinstance(node, physics.RaycastSensor):
        #     origin = self._to_screen(node.origin)
        #     for vertex in node.vertices:
        #         self.draw_segment(origin, self._to_screen(vertex), (255, 0, 0, 64))

        #     self.surface.blit(self.font.render(str(node.values[0]), True, (255,255,255)), (origin[0] + 15, origin[1]+5))
        #     self.surface.blit(self.font.render(str(node.values[1]), True, (255,255,255)), (origin[0] + 15, origin[1]-5))

        # elif isinstance(node, robot.ColorPadActuator):
        #     center = self._to_screen(node.center)
        #     radius = node.radius * self.zoom

        #     self.draw_circle(center, radius, fill=(0, 255, 255, 64))

    def draw_shape(self, shape):
        if shape.color is None:
            color = (146, 229, 146)
        else:
            color = shape.color

        # if (self.selected_shape is not None) and (shape.object_id == self.selected_shape.object_id):
        #     fill = (color[0], color[1], color[2], 87)
        #     border = (color[0], color[1], color[2], 242)
        # else:
        #     fill = (color[0], color[1], color[2], 47)
        #     border = (color[0], color[1], color[2], 142)

        fill = (color[0], color[1], color[2], 87)
        border = (color[0], color[1], color[2], 242)

        if isinstance(shape, physics.PolygonShape):
            vertices = [ self._to_screen(vertex) for vertex in shape.vertices ]
            self.draw_polygon(vertices, fill=fill, border=border)

        elif isinstance(shape, physics.CircleShape):
            center = self._to_screen(shape.center)
            radius = shape.radius * self.zoom

            self.draw_circle(center, radius, orientation=shape.orientation, fill=fill, border=border)

    def draw_segment(self, p1, p2, fill=None):
        if not fill:
            return

        pygame.draw.line(self.surface, fill, p1, p2)

    def draw_polygon(self, vertices, fill=None, border=None):
        if not vertices:
            return

        if not fill and not border:
            return

        if len(vertices) == 2:
            if border:
                pygame.draw.line(self.surface, border, vertices[0], vertices[1])
            else:
                pygame.draw.line(self.surface, fill, vertices[0], vertices[1])
        else:
            if fill:
                pygame.draw.polygon(self.surface, fill, vertices, 0)
            if border:
                pygame.draw.polygon(self.surface, border, vertices, 1)

    def draw_circle(self, center, radius, orientation=None, fill=None, border=None):
        if radius < 1: radius = 1
        else: radius = int(radius)

        if not fill and not border:
            return

        if fill:
            pygame.draw.circle(self.surface, fill, center, radius, 0)

        if border:
            pygame.draw.circle(self.surface, border, center, radius, 1)

            if orientation:
                pygame.draw.line(self.surface, border, center,
                        (center[0] + radius*orientation[0], center[1] - radius*orientation[1]))

    def draw_segment_shape(self, shape, transform, fill=None):
        v1 = self._to_screen(Box2D.b2Mul(transform, shape.vertex1))
        v2 = self._to_screen(Box2D.b2Mul(transform, shape.vertex2))

        self.draw_segment(v1, v2, fill=fill)

    def draw_polygon_shape(self, shape, transform, fill=None, border=None):
        vertices = [ self._to_screen(Box2D.b2Mul(transform, v)) for v in shape.vertices ]

        self.draw_polygon(vertices, fill=fill, border=border)

    def draw_circle_shape(self, shape, transform, fill=None, border=None):
        center = self._to_screen(Box2D.b2Mul(transform, shape.pos))
        radius = shape.radius * self.zoom
        orientation = transform.R.col2

        self.draw_circle(center, radius, orientation=orientation, fill=fill, border=border)

    def write(self, text, color):
        self.surface.blit(self.font.render(text, True, color), (5, self.__write_pos))
        self.__write_pos += 15