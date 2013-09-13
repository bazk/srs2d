1# -*- coding: utf-8 -*-
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

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "12 Oct 2013"

import logging
import ctypes
import struct

__log__ = logging.getLogger(__name__)

CURRENT_VERSION = 1
BUFFER_SIZE = 4096 * 1024

OP_TYPE = '\x00'
OP_POS = '\x01'
OP_RADIUS = '\x02'
OP_ORIENTATION = '\x03'
OP_SIZE = '\x04'
TYPE_CIRCLE = '\x00'
TYPE_SQUARE = '\x01'

class Circle(object):
    def __init__(self, x, y, radius, sin, cos):
        self.x = x
        self.y = y
        self.radius = radius
        self.sin = sin
        self.cos = cos
        self.changed = True

    def update(self, x, y, radius, sin, cos):
        if (x != self.x) or (y != self.y) or (radius != self.radius) or (sin != self.sin) or (cos != self.cos):
            self.x = x
            self.y = y
            self.radius = radius
            self.sin = sin
            self.cos = cos

            self.changed = True

    def serialize(self, id):
        self.changed = False
        res = ''
        res += struct.pack('<cHc', OP_TYPE, id, TYPE_CIRCLE)
        res += struct.pack('<cHff', OP_POS, id, self.x, self.y)
        res += struct.pack('<cHf', OP_RADIUS, id, self.radius)
        res += struct.pack('<cHff', OP_ORIENTATION, id, self.sin, self.cos)
        return res

class Square(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.changed = True

    def update(self, x, y, width, height):
        if (x != self.x) or (y != self.y) or (width != self.width) or (height != self.height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

            self.changed = True

    def serialize(self, id):
        self.changed = False
        res = ''
        res += struct.pack('<cHc', OP_TYPE, id, TYPE_SQUARE)
        res += struct.pack('<cHff', OP_POS, id, self.x, self.y)
        res += struct.pack('<cHff', OP_SIZE, id, self.width, self.height)
        return res

class SaveFile(object):
    def __init__(self, fd):
        self.fd = fd
        self.buffer = ctypes.create_string_buffer(BUFFER_SIZE)
        self.keystep_counter = 0
        self.version = 0
        self.step_rate = 0
        self.objs = []

    @staticmethod
    def new(filename, version=CURRENT_VERSION, step_rate=15):
        fd = open(filename, 'w')
        save = SaveFile(fd)
        save.version = version
        save.step_rate = step_rate
        save._insert_header()
        return save

    @staticmethod
    def open(filename):
        raise NotImplemented()

    def close(self):
        if self.offset > 0:
            self.fd.write(self.buffer.raw[:self.offset])

        self.fd.close()

    def add_circle(self, x, y, radius, sin, cos):
        circle = Circle(x, y, radius, sin, cos)
        self.objs.append(circle)
        return circle

    def add_square(self, x, y, width, height):
        square = Square(x, y, width, height)
        self.objs.append(square)
        return square

    def frame(self):
        if self.keystep_counter == 0:
            self._insert_keystep()

            for i in range(len(self.objs)):
                self._write(self.objs[i].serialize(i))
        else:
            self._insert_step()

            for i in range(len(self.objs)):
                if self.objs[i].changed:
                    self._write(self.objs[i].serialize(i))

        if self.keystep_counter >= self.step_rate:
            self.keystep_counter = 0
        else:
            self.keystep_counter += 1

    def _write(self, string, escape=True):
        if (self.offset + len(string)) >= len(self.buffer):
            self.fd.write(self.buffer.raw[:self.offset])
            self.offset = 0

        for byte in string:
            if escape and (byte == '\xFF'):
                self.buffer[self.offset] = '\xFF'
                self.offset += 1

            self.buffer[self.offset] = byte
            self.offset += 1

    def _insert_header(self):
        self.offset = 0
        self.current_step = 0
        self.keystep_counter = 0
        self._write(struct.pack('<cccBB', 'S', 'R', 'S', self.version, self.step_rate), escape=False)

    def _insert_keystep(self):
        self._write(struct.pack('<cc', '\xFF', '\xF0'), escape=False)
        self._insert_stepdef()

    def _insert_step(self):
        self._write(struct.pack('<cc', '\xFF', '\xF1'), escape=False)
        self._insert_stepdef()

    def _insert_stepdef(self):
        self._write(struct.pack('<I', self.current_step))
        self.current_step += 1