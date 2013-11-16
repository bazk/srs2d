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

import sys
import argparse
import ast
import physics
import io
import pyopencl as cl

class TestSimulator(object):
    def run(self, save=None, ann_params=None, d=0.7, num_robots=10, ta=18600, tb=5400):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        simulator = physics.Simulator(context, queue, num_worlds=1, num_robots=num_robots, ta=ta, tb=tb)

        if ann_params is not None:
            pos = physics.ANNParametersArray.load(ann_params)
        else:
            pos = physics.ANNParametersArray()

        simulator.set_ann_parameters(0, pos)
        simulator.commit_ann_parameters()
        simulator.init_worlds(d)

        if save is None:
            simulator.simulate()

        else:
            save_file = io.SaveFile.new(save, step_rate=1/float(simulator.time_step))

            arena, target_areas, target_areas_radius = simulator.get_world_transforms()
            save_file.add_object('arena', io.SHAPE_RECTANGLE, x=0.0, y=0.0, width=arena[0][0], height=arena[0][1])
            save_file.add_object('target0', io.SHAPE_CIRCLE, x=target_areas[0][0], y=target_areas[0][1], radius=target_areas_radius[0][0], sin=0.0, cos=1.0)
            save_file.add_object('target1', io.SHAPE_CIRCLE, x=target_areas[0][2], y=target_areas[0][3], radius=target_areas_radius[0][1], sin=0.0, cos=1.0)

            fitene = simulator.get_individual_fitness_energy()
            sensors, actuators, hidden = simulator.get_ann_state()
            transforms, radius = simulator.get_transforms()
            robot_radius = radius[0][0]

            robot_obj = [ None for i in range(len(transforms)) ]
            for i in range(len(transforms)):
                robot_obj[i] = save_file.add_object('robot'+str(i), io.SHAPE_CIRCLE,
                    x=transforms[i][0], y=transforms[i][1], radius=robot_radius,
                    sin=transforms[i][2], cos=transforms[i][3],
                    fitness=fitene[i][0], energy=fitene[i][1],
                    sensors=sensors[i],
                    wheels0=actuators[i][0], wheels1=actuators[i][1],
                    front_led=actuators[i][2], rear_led=actuators[i][3],
                    hidden0=hidden[i][0], hidden1=hidden[i][1], hidden2=hidden[i][2])

            cur = 0
            while (cur < (ta+tb)):
                simulator.step()

                if (cur <= ta):
                    simulator.set_fitness(0)
                    simulator.set_energy(2)

                fitene = simulator.get_individual_fitness_energy()
                sensors, actuators, hidden = simulator.get_ann_state()
                transforms, radius = simulator.get_transforms()
                for i in range(len(transforms)):
                    robot_obj[i].update(
                        x=transforms[i][0], y=transforms[i][1],
                        sin=transforms[i][2], cos=transforms[i][3],
                        fitness=fitene[i][0], energy=fitene[i][1],
                        sensors=sensors[i],
                        wheels0=actuators[i][0], wheels1=actuators[i][1],
                        front_led=actuators[i][2], rear_led=actuators[i][3],
                        hidden0=hidden[i][0], hidden1=hidden[i][1], hidden2=hidden[i][2])

                save_file.frame()

                cur += 1

            save_file.close()

        print 'fitness = ', simulator.get_fitness()[0]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save the simulation to a file", metavar="FILE")
    parser.add_argument("-p", "--params", help="parameters for the neural network", metavar="ANNPARAMS")
    parser.add_argument("-d", "--distance", type=float, help="distance between target areas", default=0.7)
    parser.add_argument("-n", "--num-robots", type=int, help="number of robots in the simulation", default=10)
    parser.add_argument("--ta", type=int, help="number of timesteps without fitness avaliation", default=18600)
    parser.add_argument("--tb", type=int, help="number of timesteps with fitness avaliation", default=5400)
    args = parser.parse_args()

    if args.params is None:
        params = None
    else:
        try:
            params = ast.literal_eval(args.params)
            if not isinstance(params, dict):
                raise Exception("Not a dictionary.")

            for n in ["weights", "bias", "weights_hidden", "bias_hidden", "timec_hidden"]:
                if not n in params:
                    raise Exception("Missing key: %s." % n)

                if not isinstance(params[n], list):
                    raise Exception("Key %s is not a list." % n)

        except Exception as e:
            print "Invalid parameters for the neural network."
            print e
            sys.exit(1)

    TestSimulator().run(args.save, params, args.distance, args.num_robots, args.ta, args.tb)