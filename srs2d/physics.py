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
__date__ = "13 Jul 2013"

import logging
import math
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

__log__ = logging.getLogger(__name__)

DEFAULT_CATEGORY_BITS = 0x0001
DEFAULT_MASK_BITS = 0xFFFF

# class Vector(object):
#     DTYPE = np.dtype([ ("x", np.float32),
#                        ("y", np.float32) ])

#     def __init__(self, x=None, y=None):
#         if x is None:
#             raise ValueError("first positional argument can not be 'None'")

#         if isinstance(x, tuple):
#             self.x, self.y = x
#         else:
#             if y is None:
#                 raise ValueError("parameter 'y' can not be 'None'")

#             self.x = x
#             self.y = y

#     def __str__(self):
#         return "Vector(%.2f, %.2f)" % (self.x, self.y)

#     def dot(self, other):
#         if isinstance(other, Vector):
#             return self.x * other.x + self.y * other.y

#         raise NotImplemented

#     def __mul__(self, other):
#         if isinstance(other, int) or isinstance(other, long) or isinstance(other, float):
#             return Vector(other * self.x, other * self.y)

#         raise NotImplemented

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __add__(self, other):
#         if isinstance(other, Vector):
#             return Vector(self.x + other.x, self.y + other.y)

#         raise NotImplemented

class Rotation(object):
    def __init__(self, angle=None, sin=None, cos=None):
        if sin is not None and cos is not None:
            self.sin = sin
            self.cos = cos
        elif angle is not None:
            self.sin = math.sin(angle)
            self.cos = math.cos(angle)
        else:
            raise NotImplementedError

    def __str__(self):
        return "Rotation(sin=%.2f, cos=%.2f)" % (self.sin, self.cos)

    @property
    def angle(self):
        return math.atan2(self.sin, self.cos)

    # @property
    # def xaxis(self):
    #     return Vector(self.cos, self.sin)

    # @property
    # def yaxis(self):
    #     return Vector(self.sin, self.cos)

    # def __mul__(self, other):
    #     if isinstance(other, Rotation):
    #         sin = self.sin * other.cos + self.cos * other.sin
    #         cos = self.cos * other.cos - self.sin * other.sin
    #         return Rotation(sin, cos)

    #     if isinstance(other, Vector):
    #         return Vector(self.cos * other.x - self.sin * other.y,
    #                       self.sin * other.x + self.cos * other.y)

    #     raise NotImplemented

    # def __rmul__(self, other):
    #     return self.__mul__(other)

class Transform(object):
    def __init__(self, position, rotation):
        self.pos = position
        self.rot = rotation

    def __str__(self):
        return "Transform(pos=%s, rot=%s)" % (str(self.pos), str(self.rot))

    # def __mul__(self, other):
    #     if isinstance(other, Transform):
    #         rot = self.rot * other.rot
    #         pos = (self.rot * other.pos) + self.pos
    #         return Transform(pos, rot)

    #     if isinstance(other, Vector):
    #         x = (self.rot.cos * other.x - self.rot.sin * other.y) + self.pos.x
    #         y = (self.rot.sin * other.x + self.rot.cos * other.y) + self.pos.y
    #         return Vector(x, y)

    #     raise NotImplemented

    # def __rmul__(self, other):
    #     return self.__mul__(other)

class World(object):
    """
    Creates a 2D top-down physics World.

    Usage:

        sim = World()

        while 1:
            sim.step()
            (step_count, clock, shapes) = sim.get_state()
            # draw_the_screen(shapes)
    """


    def __init__(self):
        super(World, self).__init__()
        self.time_step = 1.0 / 15.0
        self.step_count = 0.0
        self.clock = 0.0

        self.robots = []

        # self.context = context
        # self.queue = queue

        # src = open('something.cl', 'r')
        # self.prg = cl.Program(context, src.read()).build()

        # sizeof_buf = cl.Buffer(context, 0, 4)
        # self.prg.size_of_body_t(queue, (1,), None, sizeof_buf)

        # sizeof = np.zeros(1, dtype=np.int32)
        # cl.enqueue_copy(queue, sizeof, sizeof_buf)

        # self.sizeof_body = int(sizeof[0])

        # self.body_buf = None

    # def push_to_device(self):
    #     nbodies = len(self.bodies)

    #     self.body_buf = cl.Buffer(self.context, 0, nbodies * self.sizeof_body)

    #     pos_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     rot_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     lin_vel_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     ang_vel_arr = np.empty(nbodies, dtype=np.float32)

    #     for i in range(nbodies):
    #         pos_arr[i][0], pos_arr[i][1] = self.bodies[i].transform.pos
    #         rot_arr[i][0] = self.bodies[i].transform.rot.sin
    #         rot_arr[i][1] = self.bodies[i].transform.rot.cos
    #         lin_vel_arr[i][0], lin_vel_arr[i][1] = self.bodies[i].linear_velocity
    #         ang_vel_arr[i] = self.bodies[i].angular_speed

    #     pos_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=pos_arr)
    #     rot_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=rot_arr)
    #     lin_vel_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=lin_vel_arr)
    #     ang_vel_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=ang_vel_arr)

    #     self.prg.set_bodies(self.queue, (nbodies,), None, self.body_buf, pos_buf, rot_buf, lin_vel_buf, ang_vel_buf)

    # def pull_from_device(self):
    #     nbodies = len(self.bodies)

    #     self.body_buf = cl.Buffer(self.context, 0, nbodies * self.sizeof_body)

    #     pos_buf = cl.Buffer(self.context, 0, nbodies * 8)
    #     rot_buf = cl.Buffer(self.context, 0, nbodies * 8)
    #     lin_vel_buf = cl.Buffer(self.context, 0, nbodies * 8)
    #     ang_vel_buf = cl.Buffer(self.context, 0, nbodies * 4)

    #     self.prg.get_bodies(self.queue, (nbodies,), None, self.body_buf, pos_buf, rot_buf, lin_vel_buf, ang_vel_buf)

    #     pos_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     rot_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     lin_vel_arr = np.empty(nbodies, dtype=np.dtype( (np.float32, (2,)) ))
    #     ang_vel_arr = np.empty(nbodies, dtype=np.float32)

    #     cl.enqueue_copy(self.queue, pos_arr, pos_buf)
    #     cl.enqueue_copy(self.queue, rot_arr, rot_buf)
    #     cl.enqueue_copy(self.queue, lin_vel_arr, lin_vel_buf)
    #     cl.enqueue_copy(self.queue, ang_vel_arr, ang_vel_buf)

    #     for i in range(nbodies):
    #         self.bodies[i].transform.pos = (pos_arr[i][0], pos_arr[i][1])
    #         self.bodies[i].transform.rot = Rotation(sin=rot_arr[i][0], cos=rot_arr[i][1])
    #         self.bodies[i].linear_velocity = (lin_vel_arr[i][0], lin_vel_arr[i][1])
    #         self.bodies[i].angular_speed = ang_vel_arr[i]

    # def step_dynamics(self):
    #     self.prg.step_dynamics(self.queue, (len(self.bodies),), None, self.body_buf)
    #     self.step_count += 1

    def create_robot(self, position=(0,0), angle=0):
        robot = Robot(position, angle)
        self.robots.append(robot)
        return robot

    def step(self, time_step):
        """Run a single physics step."""

        for robot in self.robots:
            w0, w1 = robot.wheels_angular_speed
            r = robot.wheels_radius
            l0, l1 = (w0*r, w1*r)

            robot.linear_velocity = ( robot.transform.rot.sin * (l0 + l1),
                                      robot.transform.rot.cos * (l0 + l1) )

            robot.angular_speed = (l0 - l1) / (robot.wheels_distance / 2.0)

            robot.transform.pos = ( robot.transform.pos[0] + robot.linear_velocity[0] * time_step,
                                    robot.transform.pos[1] + robot.linear_velocity[1] * time_step )

            angle = math.atan2(robot.transform.rot.sin, robot.transform.rot.cos)
            angle += robot.angular_speed * time_step
            robot.transform.rot.sin = math.sin(angle)
            robot.transform.rot.cos = math.cos(angle)

        for i in range(len(self.robots)):
            for j in range(i+1, len(self.robots)):
                r1, r2 = self.robots[i], self.robots[j]

                dist = (r1.transform.pos[0] - r2.transform.pos[0]) ** 2 + (r1.transform.pos[1] - r2.transform.pos[1]) ** 2

                if dist < ((r1.body_radius + r2.body_radius) ** 2):
                    print 'collision'

        self.step_count += 1
        self.clock += time_step

class Robot(object):
    def __init__(self, position=(0, 0), angle=0, body_radius=0.06, wheels_distance=0.0825, wheels_size=(0.017, 0.03), wheels_radius=0.02, wheels_max_angular_speed=12.56):
        self.transform = Transform(position, Rotation(angle))

        self.body_radius = body_radius

        self.wheels_distance = wheels_distance
        self.wheels_size = wheels_size
        self.wheels_radius = wheels_radius

        self.wheels_angular_speed = (0, 0)
        self.wheels_max_angular_speed = wheels_max_angular_speed

        self.linear_velocity = (0,0)
        self.angular_speed = 0

    def __str__(self):
        return "Robot(transform=%s, linear_velocity=%s, angular_speed=%.2f)" % (str(self.transform), str(self.linear_velocity), self.angular_speed)

    @staticmethod
    def rpm_to_angular_speed(rpm):
        return 2 * math.pi * (rpm / 60.0)