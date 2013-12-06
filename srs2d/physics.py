# -*- coding: utf-8 -*-
#
# This file is part of srs2d.
#
# srs2d is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# srs2d is distributed in the hope that it will be useful,
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
import numpy as np
import pyopencl as cl
import logging.config
import io

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

__dir__ = os.path.dirname(__file__)

NUM_SENSORS = 13
NUM_ACTUATORS = 4
NUM_HIDDEN = 3

ANN_PARAMS_SIZE = NUM_ACTUATORS * (NUM_SENSORS+NUM_HIDDEN) + NUM_ACTUATORS + NUM_HIDDEN * NUM_SENSORS + NUM_HIDDEN + NUM_HIDDEN

class Simulator(object):
    def __init__(self, context, queue, num_worlds=1, num_robots=9, ta=600, tb=5400, time_step=1/10.0, test=False, random_targets=True):
        self.context = context
        self.queue = queue

        self.num_worlds = num_worlds
        self.num_robots = num_robots
        self.ta = ta
        self.tb = tb
        self.time_step = time_step

        self.sizeof_world_t = self.__query_sizeof_world_t(context, queue, num_robots)

        # estimate how many work items can be executed in parallel in each work group
        # self.work_group_size = pyopencl.characterize.get_simd_group_size(self.queue.device, self.sizeof_world_t)
        self.work_group_size = self.queue.device.max_work_group_size

        if self.work_group_size < num_robots:
            self.work_items_are_worlds = True
            self.global_size = (self.num_worlds,1)
            self.local_size = (1,1)

        elif self.queue.device.type == cl.device_type.GPU:
            self.work_items_are_worlds = False
            self.global_size = (self.num_worlds,self.num_robots)
            self.local_size = (1,self.num_robots)

        else:
            self.work_items_are_worlds = True
            self.global_size = (self.num_worlds,1)
            self.local_size = (1,1)

        options = [
            '-I"%s"' % os.path.join(__dir__, 'kernels/'),
            '-DNUM_WORLDS=%d' % num_worlds,
            '-DROBOTS_PER_WORLD=%d' % num_robots,
            '-DTIME_STEP=%f' % time_step,
            '-DTA=%d' % ta,
            '-DTB=%d' % tb,
            '-DWORLDS_PER_LOCAL=%d' % self.local_size[0],
            '-DROBOTS_PER_LOCAL=%d' % self.local_size[1],
        ]

        if (test):
            options.append('-DTEST')

        if (self.work_items_are_worlds):
            options.append('-DWORK_ITEMS_ARE_WORLDS')

        if (random_targets):
            options.append('-DRANDOM_TARGET_AREAS')

        src = open(os.path.join(__dir__, 'kernels/physics.cl'), 'r')
        self.prg = cl.Program(context, src.read()).build(options=' '.join(options))

        # create worlds buffer
        self.worlds = cl.Buffer(context, 0, num_worlds * self.sizeof_world_t)

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

    def simulate(self, param_list, targets_distance=1.0, targets_angle=2.356194490192345, save_hist=False):
        if len(param_list) != self.num_worlds:
            raise Exception('Number of parameters is not equal to the number of worlds!')

        param = np.zeros((len(param_list), len(param_list[0])), np.float32)
        param[:] = param_list
        param_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=param)

        # random_positions = np.random.rand(self.num_worlds, self.num_robots, 10, 4).astype(np.float32)
        # random_positions_buf = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=random_positions)

        fitness_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * self.num_worlds))

        if save_hist:
            robot_radius_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * self.num_worlds))
            arena_size_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(8 * self.num_worlds))
            target_areas_pos_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(16 * self.num_worlds))
            target_areas_radius_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(8 * self.num_worlds))

            fitness_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * (self.ta+self.tb) * self.num_worlds * self.num_robots))
            energy_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * (self.ta+self.tb) * self.num_worlds * self.num_robots))
            transform_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(16 * (self.ta+self.tb) * self.num_worlds * self.num_robots))
            sensors_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * (self.ta+self.tb) * self.num_worlds * self.num_robots * NUM_SENSORS))
            actuators_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * (self.ta+self.tb) * self.num_worlds * self.num_robots * NUM_ACTUATORS))
            hidden_hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=(4 * (self.ta+self.tb) * self.num_worlds * self.num_robots * NUM_HIDDEN))
        else:
            robot_radius_buf = None
            arena_size_buf = None
            target_areas_pos_buf = None
            target_areas_radius_buf = None

            fitness_hist_buf = None
            energy_hist_buf = None
            transform_hist_buf = None
            sensors_hist_buf = None
            actuators_hist_buf = None
            hidden_hist_buf = None

        ranluxcltab = cl.Buffer(self.context, 0, self.num_worlds * self.num_robots * 112)
        init_ranluxcl = self.prg.init_ranluxcl
        init_ranluxcl.set_scalar_arg_dtypes((np.uint32, None))
        init_ranluxcl(self.queue, self.global_size, self.local_size, np.random.randint(0, 4294967295), ranluxcltab).wait()

        simulate = self.prg.simulate
        simulate.set_scalar_arg_dtypes((None,
                                        None, np.float32, np.float32,
                                        None, np.uint32,
                                        None,
                                        None, None,
                                        None, None,
                                        None, None, None,
                                        None, None, None,
                                        np.uint32))
        simulate(self.queue, self.global_size, self.local_size,
                 ranluxcltab,
                 self.worlds, targets_distance, targets_angle,
                 param_buf, len(param_list[0]),
                 fitness_buf,
                 robot_radius_buf, arena_size_buf,
                 target_areas_pos_buf, target_areas_radius_buf,
                 fitness_hist_buf, energy_hist_buf, transform_hist_buf,
                 sensors_hist_buf, actuators_hist_buf, hidden_hist_buf,
                 1 if save_hist else 0).wait()

        fitness = np.zeros(self.num_worlds, dtype=np.float32)
        cl.enqueue_copy(self.queue, fitness, fitness_buf)

        if save_hist:
            robot_radius = np.zeros((self.num_worlds), dtype=np.float32)
            arena_size= np.zeros((self.num_worlds, 2), dtype=np.float32)
            target_areas_pos = np.zeros((self.num_worlds, 2, 2), dtype=np.float32)
            target_areas_radius = np.zeros((self.num_worlds, 2), dtype=np.float32)

            fitness_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots), dtype=np.float32)
            energy_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots), dtype=np.float32)
            transform_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots, 4), dtype=np.float32)
            sensors_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots, NUM_SENSORS), dtype=np.float32)
            actuators_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots, NUM_ACTUATORS), dtype=np.float32)
            hidden_hist = np.zeros((self.ta+self.tb, self.num_worlds, self.num_robots, NUM_HIDDEN), dtype=np.float32)

            cl.enqueue_copy(self.queue, robot_radius, robot_radius_buf)
            cl.enqueue_copy(self.queue, arena_size, arena_size_buf)
            cl.enqueue_copy(self.queue, target_areas_pos, target_areas_pos_buf)
            cl.enqueue_copy(self.queue, target_areas_radius, target_areas_radius_buf)

            cl.enqueue_copy(self.queue, fitness_hist, fitness_hist_buf)
            cl.enqueue_copy(self.queue, energy_hist, energy_hist_buf)
            cl.enqueue_copy(self.queue, transform_hist, transform_hist_buf)
            cl.enqueue_copy(self.queue, sensors_hist, sensors_hist_buf)
            cl.enqueue_copy(self.queue, actuators_hist, actuators_hist_buf)
            cl.enqueue_copy(self.queue, hidden_hist, hidden_hist_buf)

            return fitness, (
                robot_radius, arena_size, target_areas_pos, target_areas_radius,
                fitness_hist, energy_hist, transform_hist,
                sensors_hist, actuators_hist, hidden_hist
            )

        else:
            return fitness

    def simulate_and_save(self, filename, param_list, **kwargs):
        fitness, hist = self.simulate(param_list, save_hist=True, **kwargs)

        ( robot_radius, arena_size, target_areas_pos, target_areas_radius,
          fitness_hist, energy_hist, transform_hist,
          sensors_hist, actuators_hist, hidden_hist ) = hist

        save_file = io.SaveFile.new(filename, step_rate=1/float(self.time_step))

        world = 0

        save_file.add_object('arena', io.SHAPE_RECTANGLE, x=0.0, y=0.0, width=arena_size[world][0], height=arena_size[world][1])
        save_file.add_object('target0', io.SHAPE_CIRCLE, x=target_areas_pos[world][0][0], y=target_areas_pos[world][0][1],
            radius=target_areas_radius[world][0], sin=0.0, cos=1.0)
        save_file.add_object('target1', io.SHAPE_CIRCLE, x=target_areas_pos[world][1][0], y=target_areas_pos[world][1][1],
            radius=target_areas_radius[world][1], sin=0.0, cos=1.0)

        robot_obj = [ None for rid in xrange(self.num_robots) ]
        for rid in xrange(self.num_robots):
            robot_obj[rid] = save_file.add_object('robot'+str(rid), io.SHAPE_CIRCLE,
                x=transform_hist[0][world][rid][0], y=transform_hist[0][world][rid][1], radius=robot_radius[world],
                sin=transform_hist[0][world][rid][2], cos=transform_hist[0][world][rid][3],
                fitness=fitness_hist[0][world][rid], energy=energy_hist[0][world][rid],
                actuators0=actuators_hist[0][world][rid][0],
                actuators1=actuators_hist[0][world][rid][1],
                actuators2=actuators_hist[0][world][rid][2],
                actuators3=actuators_hist[0][world][rid][3],
                camera0=sensors_hist[0][world][rid][8],
                camera1=sensors_hist[0][world][rid][9],
                camera2=sensors_hist[0][world][rid][10],
                camera3=sensors_hist[0][world][rid][11],
                hidden0=hidden_hist[0][world][rid][0],
                hidden1=hidden_hist[0][world][rid][1],
                hidden2=hidden_hist[0][world][rid][2])

        cur = 0
        while (cur < (self.ta+self.tb)):
            for rid in xrange(self.num_robots):
                robot_obj[rid].update(
                    x=transform_hist[cur][world][rid][0], y=transform_hist[cur][world][rid][1],
                    sin=transform_hist[cur][world][rid][2], cos=transform_hist[cur][world][rid][3],
                    fitness=fitness_hist[cur][world][rid], energy=energy_hist[cur][world][rid],
                    actuators0=actuators_hist[cur][world][rid][0],
                    actuators1=actuators_hist[cur][world][rid][1],
                    actuators2=actuators_hist[cur][world][rid][2],
                    actuators3=actuators_hist[cur][world][rid][3],
                    camera0=sensors_hist[cur][world][rid][8],
                    camera1=sensors_hist[cur][world][rid][9],
                    camera2=sensors_hist[cur][world][rid][10],
                    camera3=sensors_hist[cur][world][rid][11],
                    hidden0=hidden_hist[cur][world][rid][0],
                    hidden1=hidden_hist[cur][world][rid][1],
                    hidden2=hidden_hist[cur][world][rid][2])

            save_file.frame()

            cur += 1

        save_file.close()

        return fitness