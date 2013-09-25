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
import random
import copy
import math
import numpy as np
import pyopencl as cl
import pyopencl.characterize
import logging.config
import logconfig

logging.config.dictConfig(logconfig.LOGGING)

__log__ = logging.getLogger(__name__)

NUM_SENSORS = 13
NUM_ACTUATORS = 4
NUM_HIDDEN = 3

class Simulator(object):
    def __init__(self, context, queue, num_worlds=1, num_robots=9, ta=600, tb=5400, time_step=1/10.0, dynamics_iterations=4):
        global NUM_INPUTS, NUM_OUTPUTS

        self.step_count = 0.0
        self.clock = 0.0

        self.context = context
        self.queue = queue

        self.num_worlds = num_worlds
        self.num_robots = num_robots
        self.ta = ta
        self.tb = tb
        self.time_step = time_step
        self.dynamics_iterations = dynamics_iterations

        options = '-DROBOTS_PER_WORLD=%d -DTIME_STEP=%f -DTA=%f -DTB=%f -DDYNAMICS_ITERATIONS=%d' % (num_robots, time_step, ta, tb, dynamics_iterations)

        src = open(os.path.join(os.path.dirname(__file__), 'kernels/physics.cl'), 'r')
        self.prg = cl.Program(context, src.read()).build(options=options)

        # query the structs sizes
        sizeof = np.zeros(1, dtype=np.int32)
        sizeof_buf = cl.Buffer(context, 0, 4)

        self.prg.size_of_world_t(queue, (1,), None, sizeof_buf).wait()
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        sizeof_world_t = int(sizeof[0])

        # estimate how many work items can be executed in parallel in each work group
        self.work_group_size = pyopencl.characterize.get_simd_group_size(self.queue.device, sizeof_world_t)
        self.global_size = (num_worlds, num_robots)
        if self.work_group_size >= num_robots:
            self.local_size = (self.work_group_size / num_robots, num_robots)
            self.need_global_barrier = False
        else:
            self.local_size = (1, self.work_group_size)
            self.need_global_barrier = True

        # create buffers
        self.worlds = cl.Buffer(context, 0, num_worlds * sizeof_world_t)

        # initialize random number generator
        self.ranluxcl = cl.Buffer(context, 0, num_worlds * num_robots * 112)
        kernel = self.prg.init_ranluxcl
        kernel.set_scalar_arg_dtypes((np.uint32, None))
        kernel(queue, self.global_size, self.local_size, random.randint(0, 4294967295), self.ranluxcl).wait()

        # initialize neural network
        self.weights = np.random.rand(num_worlds*NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)).astype(np.float32) * 10 - 5
        self.weights_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weights)
        self.bias = np.random.rand(num_worlds*NUM_ACTUATORS).astype(np.float32) * 10 - 5
        self.bias_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.bias)

        self.weights_hidden = np.random.rand(num_worlds*NUM_HIDDEN*NUM_SENSORS).astype(np.float32) * 10 - 5
        self.weights_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weights_hidden)

        self.bias_hidden = np.random.rand(num_worlds*NUM_HIDDEN).astype(np.float32) * 10 - 5
        self.bias_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.bias_hidden)

        self.timec_hidden = np.random.rand(num_worlds*NUM_HIDDEN).astype(np.float32)
        self.timec_hidden_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.timec_hidden)

        self.prg.set_ann_parameters(queue, (num_worlds,), None,
            self.ranluxcl, self.worlds, self.weights_buf, self.bias_buf,
            self.weights_hidden_buf, self.bias_hidden_buf, self.timec_hidden_buf).wait()

    def init_worlds(self, target_areas_distance):
        init_worlds = self.prg.init_worlds
        init_worlds.set_scalar_arg_dtypes((None, None, np.float32))
        init_worlds(self.queue, (self.num_worlds,), None, self.ranluxcl, self.worlds, target_areas_distance).wait()
        self.prg.init_robots(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()

    def step(self):
        if not self.need_global_barrier:
            self.prg.step_robots(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()

        else:
            self.prg.step_actuators(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds)
            self.prg.step_dynamics(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()
            self.prg.step_sensors(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds)
            self.prg.step_controllers(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()

        self.step_count += 1
        self.clock += self.time_step

    def simulate(self):
        if not self.need_global_barrier:
            self.prg.simulate(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()

        else:
            cur = 0
            while (cur < (self.ta+self.tb)):
                self.step()

                if (cur <= self.ta):
                    self.set_fitness(0)
                    self.set_energy(2)

                cur += 1

    def get_world_transforms(self):
        arena = np.zeros(self.num_worlds, dtype=np.dtype((np.float32, (2,))))
        target_areas = np.zeros(self.num_worlds, dtype=np.dtype((np.float32, (4,))))
        target_areas_radius = np.zeros(self.num_worlds, dtype=np.dtype((np.float32, (2,))))

        arena_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=arena)
        target_areas_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=target_areas)
        target_areas_radius_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=target_areas_radius)

        self.prg.get_world_transforms(self.queue, (self.num_worlds,), None, self.worlds, arena_buf, target_areas_buf, target_areas_radius_buf).wait()

        cl.enqueue_copy(self.queue, arena, arena_buf)
        cl.enqueue_copy(self.queue, target_areas, target_areas_buf)
        cl.enqueue_copy(self.queue, target_areas_radius, target_areas_radius_buf)
        return (arena, target_areas, target_areas_radius)

    def get_transforms(self):
        transforms = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (4,))))
        radius = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (1,))))
        trans_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=transforms)
        radius_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=radius)
        self.prg.get_transform_matrices(self.queue, self.global_size, self.local_size, self.worlds, trans_buf, radius_buf).wait()
        cl.enqueue_copy(self.queue, transforms, trans_buf)
        cl.enqueue_copy(self.queue, radius, radius_buf)
        return transforms, radius

    def set_ann_parameters(self, world, parameters):
        self.weights[world*NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN):(world+1)*NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)] = parameters.weights
        self.bias[world*NUM_ACTUATORS:(world+1)*NUM_ACTUATORS] = parameters.bias
        self.weights_hidden[world*NUM_HIDDEN*NUM_SENSORS:(world+1)*NUM_HIDDEN*NUM_SENSORS] = parameters.weights_hidden
        self.bias_hidden[world*NUM_HIDDEN:(world+1)*NUM_HIDDEN] = parameters.bias_hidden
        self.timec_hidden[world*NUM_HIDDEN:(world+1)*NUM_HIDDEN] = parameters.timec_hidden

    def commit_ann_parameters(self):
        cl.enqueue_copy(self.queue, self.weights_buf, self.weights)
        cl.enqueue_copy(self.queue, self.bias_buf, self.bias)
        cl.enqueue_copy(self.queue, self.weights_hidden_buf, self.weights_hidden)
        cl.enqueue_copy(self.queue, self.bias_hidden_buf, self.bias_hidden)
        cl.enqueue_copy(self.queue, self.timec_hidden_buf, self.timec_hidden)

        self.prg.set_ann_parameters(self.queue, (self.num_worlds,), None,
            self.ranluxcl, self.worlds, self.weights_buf, self.bias_buf,
            self.weights_hidden_buf, self.bias_hidden_buf, self.timec_hidden_buf).wait()

    def get_fitness(self):
        fitness = np.zeros(self.num_worlds, dtype=np.float32)
        fitness_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=fitness)
        self.prg.get_fitness(self.queue, (self.num_worlds,), None, self.worlds, fitness_buf).wait()
        cl.enqueue_copy(self.queue, fitness, fitness_buf)
        return fitness

    def set_fitness(self, fitness):
        kernel = self.prg.set_fitness
        kernel.set_scalar_arg_dtypes((None, np.float32))
        kernel(self.queue, self.global_size, self.local_size, self.worlds, fitness).wait()

    def get_individual_fitness_energy(self):
        fitene = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (2,))))
        fitene_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=fitene)
        self.prg.get_individual_fitness_energy(self.queue, self.global_size, self.local_size, self.worlds, fitene_buf).wait()
        cl.enqueue_copy(self.queue, fitene, fitene_buf)
        return fitene

    def set_energy(self, energy):
        kernel = self.prg.set_energy
        kernel.set_scalar_arg_dtypes((None, np.float32))
        kernel(self.queue, self.global_size, self.local_size, self.worlds, energy).wait()

class ANNParametersArray(object):
    def __init__(self, randomize=False, weights_boundary=(-5.0, 5.0), bias_boundary=(-5.0, 5.0), timec_boundary=(0, 1.0)):
        self.weights_boundary = weights_boundary
        self.bias_boundary = bias_boundary
        self.timec_boundary = timec_boundary

        if randomize:
            self.weights = np.random.uniform(weights_boundary[0], weights_boundary[1], NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN))
            self.bias = np.random.uniform(bias_boundary[0], bias_boundary[1], NUM_ACTUATORS)
            self.weights_hidden = np.random.uniform(weights_boundary[0], weights_boundary[1], NUM_HIDDEN * NUM_SENSORS)
            self.bias_hidden = np.random.uniform(bias_boundary[0], bias_boundary[1], NUM_HIDDEN)
            self.timec_hidden = np.random.uniform(timec_boundary[0], timec_boundary[1], NUM_HIDDEN)

        else:
            self.weights = np.zeros(NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN), dtype=np.float32)
            self.bias = np.zeros(NUM_ACTUATORS, dtype=np.float32)
            self.weights_hidden = np.zeros(NUM_HIDDEN * NUM_SENSORS, dtype=np.float32)
            self.bias_hidden = np.zeros(NUM_HIDDEN, dtype=np.float32)
            self.timec_hidden = np.zeros(NUM_HIDDEN, dtype=np.float32)

    def __str__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.weights) + len(self.bias) + len(self.weights_hidden) + \
                len(self.bias_hidden) + len(self.timec_hidden)

    def to_dict(self):
        return {
            'weights': [ x for x in self.weights.flat ],
            'bias': [ x for x in self.bias.flat ],
            'weights_hidden': [ x for x in self.weights_hidden.flat ],
            'bias_hidden': [ x for x in self.bias_hidden.flat ],
            'timec_hidden': [ x for x in self.timec_hidden.flat ]
        }

    def copy(self):
        pv = ANNParametersArray()
        pv.weights_boundary = copy.deepcopy(self.weights_boundary)
        pv.bias_boundary = copy.deepcopy(self.bias_boundary)
        pv.timec_boundary = copy.deepcopy(self.timec_boundary)
        pv.weights = np.copy(self.weights)
        pv.bias = np.copy(self.bias)
        pv.weights_hidden = np.copy(self.weights_hidden)
        pv.bias_hidden = np.copy(self.bias_hidden)
        pv.timec_hidden = np.copy(self.timec_hidden)
        return pv

    def __add__(self, other):
        if isinstance(other, ANNParametersArray):
            ret = ANNParametersArray()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights + other.weights
            ret.bias = self.bias + other.bias
            ret.weights_hidden = self.weights_hidden + other.weights_hidden
            ret.bias_hidden = self.bias_hidden + other.bias_hidden
            ret.timec_hidden = self.timec_hidden + other.timec_hidden

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __sub__(self, other):
        if isinstance(other, ANNParametersArray):
            ret = ANNParametersArray()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights - other.weights
            ret.bias = self.bias - other.bias
            ret.weights_hidden = self.weights_hidden - other.weights_hidden
            ret.bias_hidden = self.bias_hidden - other.bias_hidden
            ret.timec_hidden = self.timec_hidden - other.timec_hidden

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            ret = ANNParametersArray()
            ret.weights_boundary = self.weights_boundary
            ret.bias_boundary = self.bias_boundary
            ret.timec_boundary = self.timec_boundary

            ret.weights = self.weights * other
            ret.bias = self.bias * other
            ret.weights_hidden = self.weights_hidden * other
            ret.bias_hidden = self.bias_hidden * other
            ret.timec_hidden = self.timec_hidden * other

            self.check_boundary(self.weights_boundary, ret.weights)
            self.check_boundary(self.bias_boundary, ret.bias)
            self.check_boundary(self.weights_boundary, ret.weights_hidden)
            self.check_boundary(self.bias_boundary, ret.bias_hidden)
            self.check_boundary(self.timec_boundary, ret.timec_hidden)
            return ret
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, long):
            return self.__mul__(other)
        else:
            raise NotImplemented

    @staticmethod
    def check_boundary(boundary, array):
        for i in range(array.size):
            if array[i] < boundary[0]:
                array[i] = boundary[0]
            elif array[i] > boundary[1]:
                array[i] = boundary[1]

    def export(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'weights_hidden': self.weights_hidden,
            'bias_hidden': self.bias_hidden,
            'timec_hidden': self.timec_hidden
        }

    @staticmethod
    def load(data):
        self = ANNParametersArray()
        self.weights = np.array(data['weights'])
        self.bias = np.array(data['bias'])
        self.weights_hidden = np.array(data['weights_hidden'])
        self.bias_hidden = np.array(data['bias_hidden'])
        self.timec_hidden = np.array(data['timec_hidden'])
        return self

    def merge(self, point, other):
        if len(self) != len(other):
            raise Exception('Cannot merge arrays of different sizes.')

        if (point < 0) or (point > len(self) - 1):
            raise Exception('Point out of bounds.')

        if point < len(self.weights):
            for i in xrange(point, len(self.weights)):
                self.weights[i] = other.weights[i]
            point = 0
        else:
            point -= len(self.weights)

        if point < len(self.bias):
            for i in xrange(point, len(self.bias)):
                self.bias[i] = other.bias[i]
            point = 0
        else:
            point -= len(self.bias)

        if point < len(self.weights_hidden):
            for i in xrange(point, len(self.weights_hidden)):
                self.weights_hidden[i] = other.weights_hidden[i]
            point = 0
        else:
            point -= len(self.weights_hidden)

        if point < len(self.bias_hidden):
            for i in xrange(point, len(self.bias_hidden)):
                self.bias_hidden[i] = other.bias_hidden[i]
            point = 0
        else:
            point -= len(self.bias_hidden)

        if point < len(self.timec_hidden):
            for i in xrange(point, len(self.timec_hidden)):
                self.timec_hidden[i] = other.timec_hidden[i]
            point = 0
        else:
            point -= len(self.timec_hidden)