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

import math
import logging
import pygame
import gtk

import dropdown

__log__ = logging.getLogger(__name__)

class Scene(object):
    do_exit = False
    exit_callback = None

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
                __log__.warn("Unable to load default font or 'freesansbold.ttf', disabling text.")
                self.write = lambda *args: 0

        pygame.display.set_caption('SRS2d Viewer')

        self.screen = pygame.display.set_mode(self.resolution)
        self.surface = pygame.Surface(self.resolution, pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

        self.dropdown = dropdown.DropDown(self.font)
        #self.dropdown.add_item('Add robot', callback=self.add_robot)

        self.mouse_pos = (0,0)

        self.zoom = 180
        self.center = (0, 0)
        self.offset = (-self.resolution[0]/2, -self.resolution[1]/2)

        self.transforms = None
        self.real_clock = None
        self.speed = None

    def on_mouse_down(self, event):
        if self.dropdown.on_mouse_down(event):
            return # event handled by dropdown

    def on_mouse_up(self, event):
        if self.dropdown.on_mouse_up(event):
            return # event handled by dropdown

    def on_mouse_move(self, event):
        self.dropdown.on_mouse_move(event)

        self.mouse_pos = (event.x, event.y)

    def on_mouse_scroll(self, event):
        if event.direction == gtk.gdk.SCROLL_UP:
            self.zoom *= 1.1
        elif event.direction == gtk.gdk.SCROLL_DOWN:
            self.zoom *= 0.9

    def draw(self):
        self.screen.fill(self.background)
        self.surface.fill(self.background)
        self.__write_pos = 30

        if self.transforms is not None:
            transform = self.transforms[0]
            center = self._to_screen((transform[0], transform[1]))
            radius = 0.06 * self.zoom
            orientation = (transform[2], transform[3])
            self.draw_circle(center, radius, orientation=orientation, fill=(255,255,0,128), border=(255,255,0,255))

            for transform in self.transforms[1:]:
                center = self._to_screen((transform[0], transform[1]))
                radius = 0.06 * self.zoom
                orientation = (transform[2], transform[3])
                self.draw_circle(center, radius, orientation=orientation, fill=(255,0,0,128), border=(255,0,0,255))

            if len(self.transforms) > 1:
                orig = (self.transforms[0][0], self.transforms[0][1])
                dest = self.transform_mul_vec(self.transforms[1], (0.07, 0))

                ao = math.atan2(self.transforms[0][2], self.transforms[0][3])
                ad = math.atan2(dest[1] - orig[1], dest[0] - orig[0])

                if abs(ao - ad) <= math.radians(36):
                    self.draw_segment(self._to_screen(orig), self._to_screen(dest), (0,0,255))

                dest2 = self.transform_mul_vec(self.transforms[1], (-0.07, 0))
                ad2 = math.atan2(dest2[1] - orig[1], dest2[0] - orig[0])

                if abs(ao - ad2) <= math.radians(36):
                    self.draw_segment(self._to_screen(orig), self._to_screen(dest2), (0,255,255))

        if self.real_clock is not None:
            self.write("%02d:%02d:%02d" % (int(self.real_clock) / 3600,
                                          (int(self.real_clock) % 3600) / 60,
                                           int(self.real_clock) % 60), (80,80,255))
        if self.speed is not None:
            self.write("speed = %dx" % self.speed, (80,196,255))

        self.write(str(self.clock.get_fps()), (255,80,80))

        self.dropdown.draw(self.surface)

        self.screen.blit(self.surface, (0, 0))
        self.clock.tick()
        pygame.display.flip()

    def _to_screen(self, point):
        """Transform a world point to screen point."""

        x = ((point[0] + self.center[0]) * self.zoom) - self.offset[0]
        y = ((point[1] + self.center[1]) * self.zoom) - self.offset[1]
        y = self.resolution[1] - y

        return (int(x), int(y))

    def _to_world(self, point):
        """Transform a screen point to world point."""

        (x, y) = point
        y = self.resolution[1] - y
        x = (float(x + self.offset[0]) / self.zoom) - self.center[0]
        y = (float(y + self.offset[1]) / self.zoom) - self.center[1]

        return (x, y)

    def rot_mul_vec(self, rot, vec):
        return ( rot[1] * vec[0] - rot[0] * vec[1],
                 rot[0] * vec[0] + rot[1] * vec[1] )

    def transform_mul_vec(self, transform, vec):
        return ( (transform[3] * vec[0] - transform[2] * vec[1]) + transform[0],
                 (transform[2] * vec[0] + transform[3] * vec[1]) + transform[1] )

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
                        (center[0] + radius*orientation[1], center[1] - radius*orientation[0]))

    def write(self, text, color):
        self.surface.blit(self.font.render(text, True, color), (5, self.__write_pos))
        self.__write_pos += 15
