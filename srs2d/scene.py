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
__date__ = "11 Jul 2013"

import os
import sys
import logging
import physics
import xml.etree.ElementTree
import re
import importlib

__log__ = logging.getLogger(__name__)

class Scene(object):
    def __init__(self):
        self.modules = {}
        self.world = None

def from_xml_file(filename):
    sys.path.insert(0, os.path.dirname(os.path.realpath(filename)))

    tree = xml.etree.ElementTree.parse(filename)
    return __parse_tree(tree.getroot())

def from_xml_string(string):
    root = xml.etree.ElementTree.fromstring(string)
    return __parse_tree(root)

def __parse_tree(root):
    if root.tag != 'scene':
        raise XMLParseException("root tag is '%s', should be 'scene'." % root.tag)

    world = None
    modules = {}

    for child in root:
        if child.tag == 'import':
            if 'module' in child.attrib:
                modules[child.attrib['module']] = importlib.import_module(child.attrib['module'])

            else:
                raise XMLParseException("required 'module' attribute in 'import' tag, none found")

        elif child.tag == 'world':
            world = physics.World()

            for grandchild in child:
                world.add(__parse_node(grandchild, modules))

        else:
            raise XMLParseException("unknown tag '%s'" % child.tag)

    if world is None:
        raise XMLParseException("world tag is missing")

    return world

def __parse_node(node, modules={}):
    if node.tag == 'world':
        raise XMLParseException("cannot create a world inside a world")

    if node.tag == 'body':
        return __parse_body(node, modules)

    if node.tag == 'actuator':
        return __parse_actuator(node, modules)

    if node.tag == 'sensor':
        return __parse_sensor(node, modules)

    if node.tag == 'shape':
        return __parse_shape(node, modules)

    if node.tag == 'joint':
        return __parse_joint(node, modules)

    raise XMLParseException("unknown tag '%s'" % node.tag)

def __parse_body(node, modules={}):
    if not 'type' in node.attrib:
        raise XMLParseException("'body' tag must have the 'type' attribute (static or dynamic)")

    args = {}

    if 'position' in node.attrib:
        args['position'] = __parse_vector(node.attrib['position'])

    if node.attrib['type'] == 'static':
        body = physics.StaticBody(**args)

    elif node.attrib['type'] == 'dynamic':
        body = physics.DynamicBody(**args)

    else:
        cls = __parse_type(node.attrib['type'], modules)

        if cls is None:
            raise XMLParseException("'type' attribute value must be 'static' \
              or 'dynamic' on 'body' tags (parsed '%s')" % node.attrib['type'])

        body = cls()

    for child in node:
        body.add(__parse_node(child, modules))

    return body

def __parse_shape(node, modules={}):
    if not 'type' in node.attrib:
        raise XMLParseException("'shape' tag must have the 'type' attribute")

    args = {}

    if 'density' in node.attrib:
        args['density'] = float(node.attrib['density'])
    if 'color' in node.attrib:
        args['color'] = __parse_tuple(node.attrib['color'])

    if node.attrib['type'] == 'polygon':
        if 'vertices' in node.attrib:
            args['vertices'] = __parse_vector_list(node.attrib['vertices'])

        shape = physics.PolygonShape(**args)

    elif node.attrib['type'] == 'circle':
        if 'radius' in node.attrib:
            args['radius'] = float(node.attrib['radius'])

        shape = physics.CircleShape(**args)

    else:
        cls = __parse_type(node.attrib['type'], modules)

        if cls is None:
            raise XMLParseException("unknown 'type' attribute value '%s' \
                for 'shape' tag" % node.attrib['type'])

        shape = cls(**args)

    for child in node:
        shape.add(__parse_node(child, modules))

    return shape

def __parse_joint(node, modules={}):
    if not 'type' in node.attrib:
        raise XMLParseException("'joint' tag must have the 'type' attribute")

    args = {}

    if node.attrib['type'] == 'weld':
        del node.attrib['type']
        joint = physics.WeldJoint()

    else:
        cls = __parse_type(node.attrib['type'], modules)

        if cls is None:
            raise XMLParseException("unknown 'type' attribute value '%s' \
                for 'joint' tag" % node.attrib['type'])

        joint = cls(**args)

    for child in node:
        joint.add(__parse_node(child, modules))

    return joint

def __parse_actuator(node, modules={}):
    if not 'type' in node.attrib:
        raise XMLParseException("'actuator' tag must have the 'type' attribute")

    args = {}

    cls = __parse_type(node.attrib['type'], modules)

    if cls is None:
        raise XMLParseException("unknown 'type' attribute value '%s' \
            for 'actuator' tag" % node.attrib['type'])

    actuator = cls(**args)

    for child in node:
        actuator.add(__parse_node(child, modules))

    return actuator

def __parse_sensor(node):
    if not 'type' in node.attrib:
        raise XMLParseException("'sensor' tag must have the 'type' attribute")

    args = {}

    cls = __parse_type(node.attrib['type'], modules)

    if cls is None:
        raise XMLParseException("unknown 'type' attribute value '%s' \
            for 'sensor' tag" % node.attrib['type'])

    sensor = cls(**args)

    for child in node:
        sensor.add(__parse_node(child, modules))

    return sensor

def __parse_type(attrib, modules={}):
    for key,module in modules.iteritems():
        if attrib[:len(key)] == key:
            return module.__dict__[attrib[len(key)+1:]]

    return None

__vector_re = re.compile("^ *,? *\( *(?P<x> *[\-\+]? *[0-9]*\.?[0-9]*) *, *(?P<y> *[\-\+]? *?[0-9]*\.?[0-9]*) *\) *$")
def __parse_vector(attrib):
    matches = __vector_re.match(attrib)

    if matches is None:
        raise XMLParseException("invalid vector description '%s'" % attrib)

    return physics.Vector(float(matches.group('x')), float(matches.group('y')))

__list_re = re.compile(" *,? *\([\+\-0-9 \.,]*\) *")
def __parse_vector_list(attrib):
    matches = __list_re.findall(attrib)

    if matches is None:
        raise XMLParseException("invalid vector description '%s'" % attrib)

    return [ __parse_vector(match) for match in matches ]

class XMLParseException(Exception):
    pass