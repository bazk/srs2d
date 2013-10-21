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
import numpy

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

CURRENT_VERSION = 3
BUFFER_SIZE = 4096 * 1024

SHAPE_CIRCLE = '\xC0'
SHAPE_RECTANGLE = '\xC1'
SHAPE_POLYGON = '\xC2'

OP_CREATE_OBJ = '\xE0'
OP_CREATE_PROP = '\xE1'

OP_SET_UINT = '\xD0'
OP_SET_INT = '\xD1'
OP_SET_FLOAT = '\xD2'
OP_SET_STRING = '\xD3'

class Object(object):
    def __init__(self, id, name, shape, **kwargs):
        self.id = id
        self.name = name
        self.shape = shape
        self.properties = {}

        i = 0
        for k,v in kwargs.iteritems():
            self.properties[k] = (i, v, True)
            i += 1

    def update(self, **kwargs):
        for k,v in kwargs.iteritems():
            if not k in self.properties:
                __log__.warn("Property not set in initialization, not updating (%s)", k)
                continue

            self.properties[k] = (self.properties[k][0], v, self.properties[k][1] != v)

    def serialize_header(self):
        res = struct.pack('>cHc', OP_CREATE_OBJ, self.id, self.shape)
        res += self.name + '\0'

        for k,v in self.properties.iteritems():
            res += struct.pack('>cHH', OP_CREATE_PROP, self.id, v[0])
            res += k + '\0'

        return res

    def serialize(self, keystep=False):
        res = ''

        for k,v in self.properties.iteritems():
            if keystep or v[2]:
                if isinstance(v[1], numpy.uint32):
                    res += struct.pack('>cHHI', OP_SET_UINT, self.id, v[0], v[1])
                elif isinstance(v[1], int) or isinstance(v[1], numpy.int32):
                    res += struct.pack('>cHHi', OP_SET_INT, self.id, v[0], v[1])
                elif isinstance(v[1], float) or isinstance(v[1], numpy.float32):
                    res += struct.pack('>cHHf', OP_SET_FLOAT, self.id, v[0], v[1])
                elif isinstance(v[1], str):
                    res += struct.pack('>cHH', OP_SET_STRING, self.id, v[0])
                    res += v[1] + '\0'
                else:
                    __log__.warn("Unknow property data type, ignoring... (typeof %s is %s)", k, type(v[1]))

                self.properties[k] = (v[0], v[1], False)

        return res

class SaveFile(object):
    def __init__(self, fd):
        self.fd = fd
        self.buffer = ctypes.create_string_buffer(BUFFER_SIZE)
        self.keystep_counter = 0
        self.version = 0
        self.step_rate = 0
        self.header_written = False
        self.objs = []
        self.next_obj = 0

    @staticmethod
    def new(filename, version=CURRENT_VERSION, step_rate=15):
        fd = open(filename, 'w')
        save = SaveFile(fd)
        save.version = version
        save.step_rate = step_rate
        return save

    @staticmethod
    def open(filename):
        raise NotImplemented()

    def close(self):
        if self.offset > 0:
            self.fd.write(self.buffer.raw[:self.offset])

        self.fd.close()

    def add_object(self, name, shape, **kwargs):
        if self.header_written:
            raise Exception('Header already written!')

        obj = Object(self.next_obj, name, shape, **kwargs)
        self.next_obj += 1

        self.objs.append(obj)
        return obj

    def frame(self):
        if not self.header_written:
            self._insert_header()

        if self.keystep_counter == 0:
            self._insert_keystep()

            for obj in self.objs:
                self._write(obj.serialize(True))
        else:
            self._insert_step()

            for obj in self.objs:
                self._write(obj.serialize(False))

        if self.keystep_counter >= 100:
            self.keystep_counter = 0
        else:
            self.keystep_counter += 1

    def _insert_header(self):
        self.offset = 0
        self.current_step = 0
        self.keystep_counter = 0

        self._write(struct.pack('>cccBB', 'S', 'R', 'S', self.version, self.step_rate), escape=False)

        for i in range(len(self.objs)):
                self._write(self.objs[i].serialize_header())

        self.header_written = True

    def _insert_keystep(self):
        self._write(struct.pack('>cc', '\xFF', '\xF0'), escape=False)
        self._write(struct.pack('>I', self.current_step))
        self.current_step += 1

    def _insert_step(self):
        self._write(struct.pack('>cc', '\xFF', '\xF1'), escape=False)
        self._write(struct.pack('>I', self.current_step))
        self.current_step += 1

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