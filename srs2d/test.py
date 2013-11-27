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
    def run(self, args):
        device_type = cl.device_type.ALL
        if args.device_type == 'cpu':
            device_type = cl.device_type.CPU
        elif args.device_type == 'gpu':
            device_type = cl.device_type.GPU

        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=device_type)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context)

        simulator = physics.Simulator(context, queue, num_worlds=args.num_worlds, num_robots=args.num_robots, ta=args.ta, tb=args.tb, test=False)

        if args.params is not None:
            pos = physics.ANNParametersArray.load(args.params)
        else:
            pos = physics.ANNParametersArray()

        if args.save is None:
            fitness = simulator.simulate(args.distance, [ pos.encoded for i in xrange(args.num_worlds) ])

        else:
            fitness = simulator.simulate_and_save(args.distance, [ pos.encoded for i in xrange(args.num_worlds) ], args.save)

        print 'fitness = ', fitness[0]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-type", help="device type (all, gpu or cpu), default is all", type=str, default='all')
    parser.add_argument("-s", "--save", help="save the simulation to a file", metavar="FILE")
    parser.add_argument("-p", "--params", help="parameters for the neural network", metavar="ANNPARAMS")
    parser.add_argument("-d", "--distance", type=float, help="distance between target areas", default=0.7)
    parser.add_argument("-w", "--num-worlds", type=int, help="number of worlds in the simulation", default=120)
    parser.add_argument("-n", "--num-robots", type=int, help="number of robots in the simulation", default=10)
    parser.add_argument("--ta", type=int, help="number of timesteps without fitness avaliation", default=18600)
    parser.add_argument("--tb", type=int, help="number of timesteps with fitness avaliation", default=5400)
    args = parser.parse_args()

    TestSimulator().run(args)