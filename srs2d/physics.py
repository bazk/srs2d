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

__log__ = logging.getLogger(__name__)

NUM_SENSORS = 13
NUM_ACTUATORS = 4
NUM_HIDDEN = 3

__dir__ = os.path.dirname(__file__)

class Simulator(object):
    @staticmethod
    def test_raycast():
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        options = '-DROBOTS_PER_WORLD=%d -DTIME_STEP=%f -DTA=%f -DTB=%f -DDYNAMICS_ITERATIONS=%d' % (3, 1/10.0, 600, 5400, 4)

        src = open(os.path.join(__dir__, 'kernels/physics.cl'), 'r')
        prg = cl.Program(context, src.read()).build(options=options)

        # query the structs sizes
        sizeof = np.zeros(1, dtype=np.int32)
        sizeof_buf = cl.Buffer(context, 0, 4)

        prg.size_of_world_t(queue, (1,), None, sizeof_buf).wait()
        cl.enqueue_copy(queue, sizeof, sizeof_buf)
        sizeof_world_t = int(sizeof[0])

        # create buffers
        worlds = cl.Buffer(context, 0, sizeof_world_t)

        results = np.zeros(4, dtype=np.int32)
        results_buf = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=results)

        prg.test_raycast(queue, (1,), None, worlds, results_buf).wait()
        cl.enqueue_copy(queue, results, results_buf)

        print results

    def __init__(self, context, queue, num_worlds=1, num_robots=9, ta=600, tb=5400, time_step=1/10.0):
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

        options = '-DROBOTS_PER_WORLD=%d -DTIME_STEP=%f -DTA=%d -DTB=%d -I"%s"' % (num_robots, time_step, ta, tb, os.path.join(__dir__, 'kernels/'))

        src = open(os.path.join(__dir__, 'kernels/physics.cl'), 'r')
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
        p = parameters.decode()
        self.weights[world*NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN):(world+1)*NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)] = p['weights']
        self.bias[world*NUM_ACTUATORS:(world+1)*NUM_ACTUATORS] = p['bias']
        self.weights_hidden[world*NUM_HIDDEN*NUM_SENSORS:(world+1)*NUM_HIDDEN*NUM_SENSORS] = p['weights_hidden']
        self.bias_hidden[world*NUM_HIDDEN:(world+1)*NUM_HIDDEN] = p['bias_hidden']
        self.timec_hidden[world*NUM_HIDDEN:(world+1)*NUM_HIDDEN] = p['timec_hidden']

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

    def get_ann_state(self):
        sensors = np.zeros(self.num_worlds * self.num_robots, dtype=np.uint32)
        actuators = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (4,))))
        hidden = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (4,))))

        sensors_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=sensors)
        actuators_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=actuators)
        hidden_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=hidden)

        self.prg.get_ann_state(self.queue, self.global_size, self.local_size, self.worlds, sensors_buf, actuators_buf, hidden_buf).wait()

        cl.enqueue_copy(self.queue, sensors, sensors_buf)
        cl.enqueue_copy(self.queue, actuators, actuators_buf)
        cl.enqueue_copy(self.queue, hidden, hidden_buf)

        return sensors, actuators, hidden

class ANNParametersArray(object):
    WEIGHTS_BOUNDARY = (-5.0, 5.0)
    BIAS_BOUNDARY = (-5.0, 5.0)
    TIMEC_BOUNDARY = (0, 1.0)

    def __init__(self):
        length = (NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN) + # weights
                  NUM_ACTUATORS + # bias \
                  NUM_HIDDEN * NUM_SENSORS + # weights_hidden
                  NUM_HIDDEN + # bias_hidden
                  NUM_HIDDEN) # timec_hidden

        self.encoded = ''
        for i in xrange(length):
            self.encoded += chr(random.randint(0,254))

    def copy(self):
        n = ANNParametersArray()
        n.encoded = copy.deepcopy(self.encoded)
        return n

    def __str__(self):
        ret = ''
        for c in self.encoded:
            ret += c.encode('hex')
        return ret

    def __len__(self):
        return len(self.encoded) * 8

    def export(self):
        return self.__str__()

    @staticmethod
    def load(data):
        self = ANNParametersArray()
        self.encoded = data.decode('hex')
        return self

    def decode(self):
        ret = {
            'weights': np.zeros(NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN), dtype=np.float32),
            'bias': np.zeros(NUM_ACTUATORS, dtype=np.float32),
            'weights_hidden': np.zeros(NUM_HIDDEN * NUM_SENSORS, dtype=np.float32),
            'bias_hidden': np.zeros(NUM_HIDDEN, dtype=np.float32),
            'timec_hidden': np.zeros(NUM_HIDDEN, dtype=np.float32)
        }

        pos = 0

        for i in xrange(NUM_ACTUATORS * (NUM_SENSORS + NUM_HIDDEN)):
            ret['weights'][i] = self._decode_value(self.encoded[pos], self.WEIGHTS_BOUNDARY)
            pos += 1

        for i in xrange(NUM_ACTUATORS):
            ret['bias'][i] = self._decode_value(self.encoded[pos], self.BIAS_BOUNDARY)
            pos += 1

        for i in xrange(NUM_HIDDEN * NUM_SENSORS):
            ret['weights_hidden'][i] = self._decode_value(self.encoded[pos], self.WEIGHTS_BOUNDARY)
            pos += 1

        for i in xrange(NUM_HIDDEN):
            ret['bias_hidden'][i] = self._decode_value(self.encoded[pos], self.BIAS_BOUNDARY)
            pos += 1

        for i in xrange(NUM_HIDDEN):
            ret['timec_hidden'][i] = self._decode_value(self.encoded[pos], self.TIMEC_BOUNDARY)
            pos += 1

        return ret

    def _decode_value(self, value, boundary):
        return float(ord(value) * (boundary[1] - boundary[0])) / 255.0 + boundary[0]

    def merge(self, point, other):
        if len(self) != len(other):
            raise Exception('Cannot merge arrays of different sizes.')

        if (point < 0) or (point > (len(self) - 1)):
            raise Exception('Point out of bounds.')

        idx = int(math.floor(float(point) / 8.0))
        bit = 7 - (point % 8)  # big endian

        new = self.encoded[:idx]

        sc = ord(self.encoded[idx])
        oc = ord(other.encoded[idx])
        r = 0
        for i in range(8):
            if (i < bit):
                r |= sc & (2 ** i)
            else:
                r |= oc & (2 ** i)
        new += chr(r)

        new += other.encoded[(idx+1):]

        self.encoded = new

    def flip(self, point):
        if (point < 0) or (point > (len(self) - 1)):
            raise Exception('Point out of bounds.')

        idx = int(math.floor(float(point) / 8.0))
        bit = 7 - (point % 8)  # big endian

        c = ord(self.encoded[idx])
        if (c & (2**bit) != 0):
            c &= ~ (2**bit)
        else:
            c |= 2**bit

        self.encoded = self.encoded[:idx] + chr(c) + self.encoded[(idx+1):]