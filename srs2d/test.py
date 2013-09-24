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
__date__ = "04 Jul 2013"

import physics
import io
import pyopencl as cl

SIMULATION_DURATION = 600
NUM_ROBOTS = 10
D = 0.7

class TestSimulator(object):
    def run(self):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        simulator = physics.Simulator(context, queue, num_worlds=1, num_robots=NUM_ROBOTS, ta=600, tb=5400)

        pos = physics.ANNParametersArray(True)
        simulator.set_ann_parameters(0, pos)
        simulator.commit_ann_parameters()
        simulator.init_worlds(D)

        simulator.simulate()

        print 'fitness = ', simulator.get_fitness()[0]


    def run_and_save(self, ta=18600, tb=5400):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        simulator = physics.Simulator(context, queue, num_worlds=1, num_robots=NUM_ROBOTS, ta=ta, tb=tb)

        save = io.SaveFile.new('/tmp/test.srs', step_rate=1/float(simulator.time_step))

        pos = physics.ANNParametersArray(True)
        simulator.set_ann_parameters(0, pos)
        simulator.commit_ann_parameters()
        simulator.init_worlds(D)

        arena, target_areas, target_areas_radius = simulator.get_world_transforms()
        save.add_square(.0, .0, arena[0][0], arena[0][1])
        save.add_circle(target_areas[0][0], target_areas[0][1], target_areas_radius[0][0], .0, .1)
        save.add_circle(target_areas[0][2], target_areas[0][3], target_areas_radius[0][1], .0, .1)

        fitene = simulator.get_individual_fitness_energy()
        transforms, radius = simulator.get_transforms()
        robot_radius = radius[0][0]

        robot_obj = [ None for i in range(len(transforms)) ]
        for i in range(len(transforms)):
            robot_obj[i] = save.add_circle(transforms[i][0], transforms[i][1], robot_radius, transforms[i][2], transforms[i][3], opt1=fitene[i][0], opt2=fitene[i][1])

        cur = 0
        while (cur < (ta+tb)):
            simulator.step()

            if (cur <= ta):
                simulator.set_fitness(0)
                simulator.set_energy(2)

            fitene = simulator.get_individual_fitness_energy()
            transforms, radius = simulator.get_transforms()
            for i in range(len(transforms)):
                robot_obj[i].update(transforms[i][0], transforms[i][1], robot_radius, transforms[i][2], transforms[i][3], opt1=fitene[i][0], opt2=fitene[i][1])
            save.frame()

            cur += 1

        print 'fitness = ', simulator.get_fitness()[0]

        save.close()

if __name__=="__main__":
    TestSimulator().run_and_save()
