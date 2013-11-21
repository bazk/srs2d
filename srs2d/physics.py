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
    def __init__(self, context, queue, num_worlds=1, num_robots=9, ta=600, tb=5400, time_step=1/10.0, no_local=False, test=False):
        global NUM_INPUTS, NUM_OUTPUTS

        self.context = context
        self.queue = queue

        self.num_worlds = num_worlds
        self.num_robots = num_robots
        self.ta = ta
        self.tb = tb
        self.time_step = time_step

        self.sizeof_world_t = self.__query_sizeof_world_t(context, queue, num_robots)

        # estimate how many work items can be executed in parallel in each work group
        self.work_group_size = pyopencl.characterize.get_simd_group_size(self.queue.device, self.sizeof_world_t)

        if self.work_group_size >= num_robots:
            self.global_size = (num_worlds, num_robots)

            # manually defining the local size ensuring that all the robots from
            # the same world will be executed on the same work group (avoiding
            # concurrency problems)
            d1 = self.work_group_size / num_robots
            if (d1 > num_worlds):
                d1 = num_worlds
            self.local_size = (d1, num_robots)

            self.work_items_are_worlds = False

        else:
            self.global_size = (self.num_worlds,)

            # let opencl decide which local size is best (no concurrency
            # problems here)
            self.local_size = None

            self.work_items_are_worlds = True

        options = [
            '-I"%s"' % os.path.join(__dir__, 'kernels/'),
            '-DNUM_WORLDS=%d' % num_worlds,
            '-DROBOTS_PER_WORLD=%d' % num_robots,
            '-DTIME_STEP=%f' % time_step,
            '-DTA=%d' % ta,
            '-DTB=%d' % tb
        ]

        if (test):
            options.append('-DTEST')

        if (no_local):
            options.append('-DNO_LOCAL')

        if (self.work_items_are_worlds):
            options.append('-DWORK_ITEMS_ARE_WORLDS')
        else:
            options.append('-DWORLDS_PER_LOCAL=%d' % self.local_size[0])
            options.append('-DROBOTS_PER_LOCAL=%d' % self.local_size[1])

        src = open(os.path.join(__dir__, 'kernels/physics.cl'), 'r')
        self.prg = cl.Program(context, src.read()).build(options=' '.join(options))

        # initialize random number generator
        self.ranluxcl = cl.Buffer(context, 0, num_worlds * num_robots * 112)
        init_ranluxcl = self.prg.init_ranluxcl
        init_ranluxcl.set_scalar_arg_dtypes((np.uint32, None))
        init_ranluxcl(queue, self.global_size, self.local_size, random.randint(0, 4294967295), self.ranluxcl).wait()

        # create and initialize worlds
        self.worlds = cl.Buffer(context, 0, num_worlds * self.sizeof_world_t)
        self.init_worlds(0.7)

    def __query_sizeof_world_t(self, context, queue, num_robots):
        src = '''
        #include <defs.cl>

        __kernel void size_of_world_t(__global unsigned int *result)
        {
            *result = (unsigned int) sizeof(world_t);
        }
        '''

        prg = cl.Program(context, src).build(options='-I"%s" -DROBOTS_PER_WORLD=%d' % (os.path.join(__dir__, 'kernels/'), num_robots))

        sizeof_buf = cl.Buffer(context, 0, 4)
        prg.size_of_world_t(queue, (1,), None, sizeof_buf).wait()

        sizeof = np.zeros(1, dtype=np.uint32)
        cl.enqueue_copy(queue, sizeof, sizeof_buf).wait()
        return int(sizeof[0])

    def init_worlds(self, target_areas_distance):
        init_worlds = self.prg.init_worlds
        init_worlds.set_scalar_arg_dtypes((None, None, np.float32))
        init_worlds(self.queue, (self.num_worlds,), None, self.ranluxcl, self.worlds, target_areas_distance).wait()

    def step(self, current_step):
        step_robots = self.prg.step_robots
        step_robots.set_scalar_arg_dtypes((None, None, np.uint32))
        step_robots(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds, current_step).wait()

    def simulate(self):
        self.prg.simulate(self.queue, self.global_size, self.local_size, self.ranluxcl, self.worlds).wait()

    def set_ann_parameters(self, parameters):
        if len(parameters) != self.num_worlds:
            raise Exception('Number of parameters is not equal to the number of worlds!')

        param = np.chararray(len(parameters), len(parameters[0]))
        param[:] = parameters

        param_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=param)

        set_ann_parameters = self.prg.set_ann_parameters
        set_ann_parameters.set_scalar_arg_dtypes((None, None, np.uint32))
        set_ann_parameters(self.queue, (self.num_worlds,), None, self.worlds, param_buf, len(parameters[0])).wait()

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

    def get_fitness(self):
        fitness = np.zeros(self.num_worlds, dtype=np.float32)
        fitness_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=fitness)
        self.prg.get_fitness(self.queue, (self.num_worlds,), None, self.worlds, fitness_buf).wait()
        cl.enqueue_copy(self.queue, fitness, fitness_buf)
        return fitness

    def get_individual_fitness_energy(self):
        fitene = np.zeros(self.num_worlds * self.num_robots, dtype=np.dtype((np.float32, (2,))))
        fitene_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=fitene)
        self.prg.get_individual_fitness_energy(self.queue, self.global_size, self.local_size, self.worlds, fitene_buf).wait()
        cl.enqueue_copy(self.queue, fitene, fitene_buf)
        return fitene

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
        return len(self.encoded)

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

        if (point < 0) or (point > (len(self)*8 - 1)):
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
        if (point < 0) or (point > (len(self)*8 - 1)):
            raise Exception('Point out of bounds.')

        idx = int(math.floor(float(point) / 8.0))
        bit = 7 - (point % 8)  # big endian

        c = ord(self.encoded[idx])
        if (c & (2**bit) != 0):
            c &= ~ (2**bit)
        else:
            c |= 2**bit

        self.encoded = self.encoded[:idx] + chr(c) + self.encoded[(idx+1):]