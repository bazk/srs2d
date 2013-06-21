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
import logging
import threading
import pygame
import Box2D

__log__ = logging.getLogger(__name__)

class Viewer(object):
    background = (0, 0, 0)
    resolution = (1024, 768)

    _flipX = False
    _flipY = False

    exit = False

    def __init__(self):
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

        self.zoom = 180
        self.center = Box2D.b2Vec2(0, 0)
        self.offset = (-self.resolution[0]/2, -self.resolution[1]/2)

        while not self.exit:
            self.draw()

    def draw(self):
        self.screen.fill(self.background)
        self.surface.fill(self.background)
        self.__write_pos = 30
        self.check_events()

        self.draw_circle(self._to_screen(Box2D.b2Vec2(0,0)), 0.01 * self.zoom, fill=(0,0,0))

        self.write(str(self.clock.get_fps()), (200, 0, 0))

        self.screen.blit(self.surface, (0, 0))
        self.clock.tick(30)
        pygame.display.flip()

    def check_events(self):
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True

            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_q) or (event.key == pygame.K_ESCAPE):
                    self.exit = True

    def _to_screen(self, point):
        """Transform a world point to screen point."""

        x = ((point.x + self.center.x) * self.zoom) - self.offset[0]
        y = ((point.y + self.center.y) * self.zoom) - self.offset[1]

        if self._flipX:
            x = self.resolution[0] - x

        if self._flipY:
            y = self.resolution[1] - y

        return (int(x), int(y))

    def _to_world(self, point):
        """Transform a screen point to world point."""

        (x, y) = point

        if self._flipX:
            x = self.resolution[0] - x

        if self._flipY:
            y = self.resolution[1] - y

        x = (float(x + self.offset[0]) / self.zoom) - self.center.x
        y = (float(y + self.offset[1]) / self.zoom) - self.center.y

        return Box2D.b2Vec2(x, y)

    def write(self, text, color):
        self.surface.blit(self.font.render(text, True, color), (5, self.__write_pos))
        self.__write_pos += 15

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

        self.DrawSegment(v1, v2, fill=fill)

    def draw_polygon_shape(self, shape, transform, fill=None, border=None):
        vertices = [ self._to_screen(Box2D.b2Mul(transform, v)) for v in shape.vertices ]

        self.DrawPolygon(vertices, fill=fill, border=border)

    def draw_circle_shape(self, shape, transform, fill=None, border=None):
        center = self._to_screen(Box2D.b2Mul(transform, shape.pos))
        radius = shape.radius * self.zoom
        orientation = transform.R.col2

        self.DrawCircle(center, radius, orientation=orientation, fill=fill, border=border)

    def draw_shape(self, shape, transform, fill=None, border=None):
        if isinstance(shape, Box2D.b2PolygonShape):
            self.DrawPolygonShape(shape, transform, fill=fill, border=border)

        elif isinstance(shape, Box2D.b2CircleShape):
            self.DrawCircleShape(shape, transform, fill=fill, border=border)

        elif isinstance(shape, Box2D.b2EdgeShape):
            self.DrawSegmentShape(shape, transform, fill=fill)

        elif isinstance(shape, Box2D.b2LoopShape):
            vertices = shape.vertices
            v1 = self._to_screen(Box2D.b2Mul(transform, vertices[-1]))
            for v2 in vertices:
                v2 = self._to_screen(Box2D.b2Mul(transform, v2))
                self.DrawSegment(v1, v2, color)
                v1 = v2

if __name__=="__main__":
    viewer = Viewer()
