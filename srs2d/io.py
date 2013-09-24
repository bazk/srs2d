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

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "12 Oct 2013"

import logging
import ctypes
import struct

__log__ = logging.getLogger(__name__)

CURRENT_VERSION = 2
BUFFER_SIZE = 4096 * 1024

OP_TYPE = '\x00'
OP_POS = '\x01'
OP_RADIUS = '\x02'
OP_ORIENTATION = '\x03'
OP_SIZE = '\x04'
OP_OPT1 = '\x05'
OP_OPT2 = '\x06'
TYPE_CIRCLE = '\x00'
TYPE_SQUARE = '\x01'

class Object(object):
    def __init__(self, **kwargs):
        self.properties = {}

        for k,v in kwargs.iteritems():
            self.properties[k] = (v, True)

    def update(self, **kwargs):
        for k,v in kwargs.iteritems():
            if not k in self.properties:
                __log__.warn("Property not set in initialization, not updating (%s)", k)
                continue

            self.properties[k] = (v, self.properties[k][0] != v)

    def set_unchanged(self):
        for k,v in self.properties.iteritems():
            self.properties[k] = (v[0], False)

class Circle(Object):
    def __init__(self, x, y, radius, sin, cos, opt1=None, opt2=None):
        super(Circle, self).__init__(x=x, y=y, radius=radius, sin=sin, cos=cos, opt1=opt1, opt2=opt2)

    def update(self, x, y, radius, sin, cos, opt1=None, opt2=None):
        super(Circle, self).update(x=x, y=y, radius=radius, sin=sin, cos=cos, opt1=opt1, opt2=opt2)

    def serialize(self, id, keystep=False):
        props = self.properties

        res = ''

        if keystep:
            res += struct.pack('>cHc', OP_TYPE, id, TYPE_CIRCLE)

        if keystep or (props['x'][1]) or (props['y'][1]):
            res += struct.pack('>cHff', OP_POS, id, props['x'][0], props['y'][0])

        if keystep or (props['radius'][1]):
            res += struct.pack('>cHf', OP_RADIUS, id, props['radius'][0])

        if keystep or (props['sin'][1]) or (props['cos'][1]):
            res += struct.pack('>cHff', OP_ORIENTATION, id, props['sin'][0], props['cos'][0])

        if ( keystep or (props['opt1'][1]) ) and (props['opt1'][0] is not None):
            res += struct.pack('>cHf', OP_OPT1, id, props['opt1'][0])

        if ( keystep or (props['opt2'][1]) ) and (props['opt2'][0] is not None):
            res += struct.pack('>cHf', OP_OPT2, id, props['opt2'][0])

        self.set_unchanged()

        return res

class Square(Object):
    def __init__(self, x, y, width, height, opt1=None, opt2=None):
        super(Square, self).__init__(x=x, y=y, width=width, height=height, opt1=opt1, opt2=opt2)

    def update(self, x, y, width, height, opt1=None, opt2=None):
        super(Square, self).update(x=x, y=y, width=width, height=height, opt1=opt1, opt2=opt2)

    def serialize(self, id, keystep=False):
        props = self.properties

        res = ''

        if keystep:
            res += struct.pack('>cHc', OP_TYPE, id, TYPE_SQUARE)

        if keystep or (props['x'][1]) or (props['y'][1]):
            res += struct.pack('>cHff', OP_POS, id, props['x'][0], props['y'][0])

        if keystep or (props['width'][1]) or (props['height'][1]):
            res += struct.pack('>cHff', OP_SIZE, id, props['width'][0], props['height'][0])

        if ( keystep or (props['opt1'][1]) ) and (props['opt1'][0] is not None):
            res += struct.pack('>cHf', OP_OPT1, id, props['opt1'][0])

        if ( keystep or (props['opt2'][1]) ) and (props['opt2'][0] is not None):
            res += struct.pack('>cHf', OP_OPT2, id, props['opt2'][0])

        self.set_unchanged()

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

    def add_circle(self, x, y, radius, sin, cos, opt1=None, opt2=None):
        circle = Circle(x, y, radius, sin, cos, opt1, opt2)
        self.objs.append(circle)
        return circle

    def add_square(self, x, y, width, height, opt1=None, opt2=None):
        square = Square(x, y, width, height, opt1, opt2)
        self.objs.append(square)
        return square

    def frame(self):
        if self.keystep_counter == 0:
            self._insert_keystep()

            for i in range(len(self.objs)):
                self._write(self.objs[i].serialize(i, True))
        else:
            self._insert_step()

            for i in range(len(self.objs)):
                self._write(self.objs[i].serialize(i, False))

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
        self._write(struct.pack('>cccBB', 'S', 'R', 'S', self.version, self.step_rate), escape=False)

    def _insert_keystep(self):
        self._write(struct.pack('>cc', '\xFF', '\xF0'), escape=False)
        self._insert_stepdef()

    def _insert_step(self):
        self._write(struct.pack('>cc', '\xFF', '\xF1'), escape=False)
        self._insert_stepdef()

    def _insert_stepdef(self):
        self._write(struct.pack('>I', self.current_step))
        self.current_step += 1
