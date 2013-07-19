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

NUM_SENSORS = 13
NUM_ACTUATORS = 4
NUM_HIDDEN = 3

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


    def __init__(self, context, queue, num_worlds=1, num_robots=9, target_areas_distance=1.2):
        global NUM_INPUTS, NUM_OUTPUTS

        self.step_count = 0.0
        self.clock = 0.0

        self.context = context
        self.queue = queue

        self.num_worlds = 1
        self.num_robots = num_robots

        src = open(os.path.join(os.path.dirname(__file__), 'kernels/physics.cl'), 'r')
        self.prg = cl.Program(context, src.read()).build(options='-D ROBOTS_PER_WORLD=%d' % num_robots)

        # query the structs sizes
        sizeof = np.zeros(1, dtype=np.int32)
        sizeof_buf = cl.Buffer(context, 0, 4)

        self.prg.size_of_robot_t(queue, (1,), None, sizeof_buf).wait()
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        sizeof_robot_t = int(sizeof[0])

        self.prg.size_of_target_area_t(queue, (1,), None, sizeof_buf).wait()
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        size_of_target_area_t = int(sizeof[0])

        # create buffers
        self.robots = cl.Buffer(context, 0, num_worlds * num_robots * sizeof_robot_t)
        self.target_areas = cl.Buffer(context, 0, 2 * size_of_target_area_t)

        # initialize random number generator
        self.ranluxcl = cl.Buffer(context, 0, num_robots * 112)
        kernel = self.prg.init_ranluxcl
        kernel.set_scalar_arg_dtypes((np.uint32, None))
        kernel(queue, (num_robots,), None, random.randint(0, 4294967295), self.ranluxcl).wait()

        # initialize robots and target_areas
        self.prg.init_robots(queue, (num_worlds, num_robots), None, self.ranluxcl, self.robots).wait()

        kernel = self.prg.init_target_areas
        kernel.set_scalar_arg_dtypes((None, None, np.float32))
        kernel(queue, (2,), None, self.ranluxcl, self.target_areas, target_areas_distance).wait()

        # initialize neural network
        self.weights = np.random.rand(NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)).astype(np.float32) * 10 - 5
        self.weights_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weights)
        self.bias = np.random.rand(NUM_ACTUATORS).astype(np.float32) * 10 - 5
        self.bias_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.bias)

        self.weights_hidden = np.random.rand(NUM_HIDDEN*NUM_SENSORS).astype(np.float32) * 10 - 5
        self.weights_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weights_hidden)
        self.bias_hidden = np.random.rand(NUM_HIDDEN).astype(np.float32) * 10 - 5
        self.bias_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.bias_hidden)
        self.timec_hidden = np.random.rand(NUM_HIDDEN).astype(np.float32)
        self.timec_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.timec_hidden)

        self.H = np.zeros(NUM_HIDDEN).astype(np.float32)
        self.H_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.H)

        self.prg.set_robots_ann(queue, (num_worlds, num_robots), None,
            self.ranluxcl, self.robots, self.weights_buf, self.bias_buf,
            self.weights_hidden_buf, self.bias_hidden_buf, self.timec_hidden_buf, self.H_buf).wait()

    def step(self, time_step=1/30.0, dynamics_iterations=4):
        kernel = self.prg.step_robots
        kernel.set_scalar_arg_dtypes((None, None, None, np.float32, np.uint32))
        kernel(self.queue, (self.num_worlds,self.num_robots), None, self.ranluxcl, self.robots, self.target_areas, time_step, dynamics_iterations).wait()

        self.step_count += 1
        self.clock += time_step

    def simulate(self, seconds, time_step=1/30.0, dynamics_iterations=4):
        kernel = self.prg.simulate
        kernel.set_scalar_arg_dtypes((None, None, None, np.float32, np.uint32, np.float32))
        kernel(self.queue, (self.num_worlds,self.num_robots), None, self.ranluxcl, self.robots, self.target_areas, time_step, dynamics_iterations, seconds).wait()

    def get_transforms(self):
        transforms = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (4,))))
        trans_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=transforms)
        self.prg.get_transform_matrices(self.queue, (self.num_worlds, self.num_robots), None, self.robots, trans_buf).wait()
        cl.enqueue_copy(self.queue, transforms, trans_buf)
        return transforms