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

import os
import logging
import math
import random
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
import ann

__log__ = logging.getLogger(__name__)

NUM_INPUTS = 13
NUM_OUTPUTS = 4

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


    def __init__(self, context, queue, num_robots=9, target_areas_distance=1.2):
        global NUM_INPUTS, NUM_OUTPUTS

        self.step_count = 0.0
        self.clock = 0.0

        self.context = context
        self.queue = queue

        self.num_robots = num_robots

        self.controllers = [ ann.NeuralNetworkController() for i in range(num_robots) ]

        src = open(os.path.join(os.path.dirname(__file__), 'physics.cl'), 'r')
        self.prg = cl.Program(context, src.read()).build()

        # query the structs sizes
        sizeof = np.zeros(1, dtype=np.int32)
        sizeof_buf = cl.Buffer(context, 0, 4)

        self.prg.size_of_robot_t(queue, (1,), None, sizeof_buf)
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        sizeof_robot_t = int(sizeof[0])

        self.prg.size_of_target_area_t(queue, (1,), None, sizeof_buf)
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        size_of_target_area_t = int(sizeof[0])

        # create buffers
        self.robots = cl.Buffer(context, 0, num_robots * sizeof_robot_t)
        self.target_areas = cl.Buffer(context, 0, 2 * size_of_target_area_t)

        self.inputs = np.zeros(num_robots * NUM_INPUTS, dtype=np.float32)
        self.in_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.inputs)
        self.outputs = np.zeros(num_robots * NUM_OUTPUTS, dtype=np.float32)
        self.out_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.outputs)

        # initialize random number generator
        self.ranluxcl = cl.Buffer(context, 0, num_robots * 112)
        kernel = self.prg.init_ranluxcl
        kernel.set_scalar_arg_dtypes((np.uint32, None))
        kernel(queue, (num_robots,), None, random.randint(0, 4294967295), self.ranluxcl)

        # initialize robots and target_areas
        self.prg.init_robots(queue, (num_robots,), None, self.ranluxcl, self.robots)
        kernel = self.prg.init_target_areas
        kernel.set_scalar_arg_dtypes((None, np.float32))
        kernel(queue, (2,), None, self.target_areas, target_areas_distance)

    def step(self, time_step):
        cl.enqueue_copy(self.queue, self.out_buf, self.outputs)
        self.prg.step_actuators(self.queue, (self.num_robots,), None, self.robots, self.out_buf)

        kernel = self.prg.step_dynamics
        kernel.set_scalar_arg_dtypes((np.float32, None))
        kernel(self.queue, (self.num_robots,), None, time_step, self.robots)

        self.prg.step_sensors(self.queue, (self.num_robots,), None, self.ranluxcl, self.robots, self.target_areas, self.in_buf)
        cl.enqueue_copy(self.queue, self.inputs, self.in_buf)

        for gid in range(self.num_robots):
            inputs = self.inputs[gid*NUM_INPUTS:gid*NUM_INPUTS+NUM_INPUTS]
            outputs = self.outputs[gid*NUM_OUTPUTS:gid*NUM_OUTPUTS+NUM_OUTPUTS]
            self.controllers[gid].think(inputs, outputs)

        self.step_count += 1
        self.clock += time_step

    def get_transforms(self):
        transforms = np.zeros(self.num_robots, dtype=np.dtype((np.float32, (4,))))
        trans_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=transforms)
        self.prg.get_transform_matrices(self.queue, (self.num_robots,), None, self.robots, trans_buf)
        cl.enqueue_copy(self.queue, transforms, trans_buf)
        return transforms