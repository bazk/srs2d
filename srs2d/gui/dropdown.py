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
Provides a drop down menu system for the viewer main window.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "21 Jun 2013"

import logging
import pygame
import math

__log__ = logging.getLogger(__name__)

class DropDown(object):
    width = 0
    height = 0

    bg = (180, 200, 202, 196)
    bg_hl = (180, 200, 202, 255)

    hidden = True
    pos = (0, 0)

    margin_top = 4
    margin_right = 4
    margin_bottom = 4
    margin_left = 4

    def __init__(self, font):
        self.font = font

        wid, hei = self.font.size('TEST')
        self.item_height = hei

        self.items = []

    def draw(self, surface):
        if self.hidden:
            return

        if len(self.items) == 0:
            return

        maxwidth = 0
        for item in self.items:
            width, height = self.font.size(item.text)
            if width > maxwidth:
                maxwidth = width

        self.width = maxwidth + self.margin_left + self.margin_right
        self.height = len(self.items) * (self.item_height + self.margin_top + self.margin_bottom)

        hl_item = self._get_highlighted_item()

        xpos = self.pos[0]
        ypos = self.pos[1]

        idx = 0

        for item in self.items:
            width, height = self.font.size(item.text)

            height += self.margin_top + self.margin_bottom

            bg = self.bg
            if item == hl_item:
                bg = self.bg_hl

            box = pygame.Rect(xpos, ypos, self.width, height)
            pygame.draw.rect(surface, bg, box, 0)

            surface.blit(self.font.render(item.text, True, (0, 0, 0)),
                    (xpos + self.margin_left, ypos + self.margin_top))

            idx += 1
            ypos += height

    def mouse_down(self, event):
        pass

    def mouse_up(self, event):
        if self.hidden:
            if event.button == 3:
                self.hidden = False
                self.pos = event.pos

        else:
            if self._is_inside(event.pos):
                if event.button == 1:
                    item = self._get_highlighted_item()
                    if item is not None:
                        item.click()

            else:
                if event.button == 3:
                    self.hidden = False
                    self.pos = event.pos
                else:
                    self.hidden = True

    def add_item(self, item):
        self.items.append(item)

    def _is_inside(self, pos):
        if (pos[0] > self.pos[0]) and (pos[0] < (self.pos[0] + self.width)) and \
           (pos[1] > self.pos[1]) and (pos[1] < (self.pos[1] + self.height)):
            return True

        return False


    def _get_highlighted_item(self):
        mouse = pygame.mouse.get_pos()

        if not self._is_inside(mouse):
            return None

        idx = int(math.floor(float(mouse[1] - self.pos[1]) /
            float(self.item_height + self.margin_top + self.margin_bottom)))

        return self.items[idx]


class DropDownItem(object):
    def __init__(self, text):
        self.text = text

    def click(self):
        pass
