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

import argparse
import physics
import pyopencl as cl

class TestSimulator(object):
    def run(self, save=None, ann_params=None, d=0.7, num_robots=10, ta=18600, tb=5400):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        num_worlds = 120

        simulator = physics.Simulator(context, queue, num_worlds=num_worlds, num_robots=num_robots, ta=ta, tb=tb, test=False)

        if ann_params is not None:
            pos = physics.ANNParametersArray.load(ann_params)
        else:
            pos = physics.ANNParametersArray()

        if save is None:
            fitness = simulator.simulate(d, [ pos.encoded for i in xrange(num_worlds) ])

        else:
            fitness = simulator.simulate_and_save(d, [ pos.encoded for i in xrange(num_worlds) ], save)

        print 'fitness = ', fitness[0]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save the simulation to a file", metavar="FILE")
    parser.add_argument("-p", "--params", help="parameters for the neural network", metavar="ANNPARAMS")
    parser.add_argument("-d", "--distance", type=float, help="distance between target areas", default=0.7)
    parser.add_argument("-n", "--num-robots", type=int, help="number of robots in the simulation", default=10)
    parser.add_argument("--ta", type=int, help="number of timesteps without fitness avaliation", default=18600)
    parser.add_argument("--tb", type=int, help="number of timesteps with fitness avaliation", default=5400)
    args = parser.parse_args()

    TestSimulator().run(args.save, args.params, args.distance, args.num_robots, args.ta, args.tb)