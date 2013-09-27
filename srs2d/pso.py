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
__date__ = "04 Jul 2013"

import os
import argparse
import random
import logging
import physics
import pyopencl as cl
import logging.config
import logconfig
import solace
import io

logging.config.dictConfig(logconfig.LOGGING)
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("-w", "--inertia",          help="set PSO inertia (W) parameter, default is 0.9", type=float, default=0.9)
    parser.add_argument("-a", "--alfa",             help="set PSO alfa parameter, default is 2.0", type=float, default=2)
    parser.add_argument("-b", "--beta",             help="set PSO beta parameter, default is 2.0", type=float, default=2)
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="PSO population size (particles), default is 10", type=int, default=10)
    parser.add_argument("-d", "--distances",        help="list of distances between target areas to be evaluated each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("-t", "--trials",           help="number of trials per distance, default is 3", type=int, default=3)
    args = parser.parse_args()

    uri = os.environ.get('SOLACE_URI')
    username = os.environ.get('SOLACE_USERNAME')
    password = os.environ.get('SOLACE_PASSWORD')

    if (uri is None) or (username is None) or (password is None):
        raise Exception('Environment variables (SOLACE_URI, SOLACE_USERNAME, SOLACE_PASSWORD) not set!')

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(args.num_runs, {
        'W': args.inertia,
        'ALFA': args.alfa,
        'BETA': args.beta,
        'STEPS_TA': args.ta,
        'STEPS_TB': args.tb,
        'NUM_GENERATIONS': args.num_generations,
        'NUM_RUNS': args.num_runs,
        'NUM_ROBOTS': args.num_robots,
        'POPULATION_SIZE': args.population_size,
        'D': args.distances,
        'TRIALS': args.trials,
    })

    for run in inst.runs:
        PSO(context, queue).execute(run, args)

class PSO(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def execute(self, run, args):
        __log__.info('Starting PSO...')

        run.begin()

        self.gbest = None
        self.gbest_fitness = None

        self.particles = [ Particle(args.inertia, args.alfa, args.beta) for i in range(args.population_size) ]

        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=args.population_size,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb)

        generation = 1

        while (generation <= args.num_generations):
            __log__.info('[gen=%d] Evaluating particles...', generation)
            self.evaluate(args.distances, args.trials)

            __log__.info('[gen=%d] Updating particles...', generation)
            for p in self.particles:
                p.update_pbest()

            new_gbest = False
            for p in self.particles:
                if (self.gbest is None) or (p.pbest.fitness > self.gbest.fitness):
                    self.gbest = p.pbest.copy()
                    new_gbest = True

            for p in self.particles:
                p.gbest = self.gbest
                p.update_pos_vel()

            __log__.info('[gen=%d] Particles updated, current gbest: %s', generation, str(self.gbest))

            if new_gbest:
                run.progress(generation / float(args.num_generations), {
                    'generation': generation,
                    'gbest_fitness': self.gbest.fitness,
                    'gbest_position': self.gbest.position.to_dict()
                })

                if not args.no_save:
                    __log__.info('[gen=%d] Saving simulation for the new found gbest...', generation)
                    fit = self.simulate_and_save('/tmp/simulation.srs', self.gbest.position, args.ta, args.tb,
                        args.num_robots, args.distances[ random.randint(0, len(args.distances)-1) ])

                    run.upload('/tmp/simulation.srs', 'run-%02d-new-gbest-gen-%04d-fit-%.5f.srs' % (run.id, generation, fit))

            else:
                run.progress(generation / float(args.num_generations), {'generation': generation})

            generation += 1

        run.done()

    def evaluate(self, distances, trials):
        for p in range(len(self.particles)):
            self.simulator.set_ann_parameters(p, self.particles[p].position)
        self.simulator.commit_ann_parameters()

        for p in range(len(self.particles)):
            self.particles[p].fitness = 0.0

        for d in distances:
            for i in range(trials):
                self.simulator.init_worlds(d)
                self.simulator.simulate()

                fit = self.simulator.get_fitness()
                for p in range(len(self.particles)):
                    self.particles[p].fitness += fit[p]

        for p in range(len(self.particles)):
            self.particles[p].fitness /= len(distances) * trials

    def simulate_and_save(self, filename, pos, ta, tb, num_robots, distance):
        simulator = physics.Simulator(self.context, self.queue,
                                      num_worlds=1,
                                      num_robots=num_robots,
                                      ta=ta, tb=tb)

        save = io.SaveFile.new(filename, step_rate=1/float(simulator.time_step))

        simulator.set_ann_parameters(0, pos)
        simulator.commit_ann_parameters()
        simulator.init_worlds(distance)

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

        current_step = 0
        while current_step < (ta + tb):
            simulator.step()

            if (current_step <= ta):
                simulator.set_fitness(0)
                simulator.set_energy(2)

            fitene = simulator.get_individual_fitness_energy()
            transforms, radius = simulator.get_transforms()
            for i in range(len(transforms)):
                robot_obj[i].update(transforms[i][0], transforms[i][1], robot_radius, transforms[i][2], transforms[i][3], opt1=fitene[i][0], opt2=fitene[i][1])
            save.frame()
            current_step += 1

        save.close()

        return simulator.get_fitness()[0]

class Particle(object):
    def __init__(self, inertia=0.9, alfa=2.0, beta=2.0):
        self.id = id(self)

        self.inertia = inertia
        self.alfa = alfa
        self.beta = beta

        self.position = physics.ANNParametersArray(True)
        self.velocity = physics.ANNParametersArray(True)
        self.fitness = 0.0

        self.pbest = None
        self.gbest = None

    def __str__(self):
        return 'Particle(%d, fitness=%.5f)' % (self.id, self.fitness)

    def copy(self):
        p = Particle(self.inertia, self.alfa, self.beta)
        p.position = self.position.copy()
        p.velocity = self.velocity.copy()
        p.fitness = self.fitness
        p.pbest = self.pbest
        p.gbest = self.gbest
        return p

    def update_pbest(self):
        if (self.pbest is None) or (self.fitness > self.pbest.fitness):
            self.pbest = self.copy()

    def update_pos_vel(self):
        self.velocity = self.inertia * self.velocity + \
                        self.alfa * random.uniform(0, 1.0) * (self.pbest.position - self.position) + \
                        self.beta * random.uniform(0, 1.0) * (self.gbest.position - self.position)

        self.position = self.position + self.velocity

if __name__=="__main__":
    main()