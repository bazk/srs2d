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
__date__ = "19 Sep 2013"

import os
import argparse
import random
import logging
import physics
import pyopencl as cl
import logging.config
import solace
import io
import png

logging.basicConfig(format='[ %(asctime)s ] [%(levelname)s] %(message)s')
__log__ = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity",        help="increase output verbosity", action="count")
    parser.add_argument("-q", "--quiet",            help="supress output (except errors)", action="store_true")
    parser.add_argument("--no-save",                help="skip saving best fitness simulation", action="store_true")
    parser.add_argument("--image",                  help="generate and upload an image representing current particle population", action="store_true")
    parser.add_argument("--ta",                     help="number of timesteps without fitness avaliation, default is 600", type=int, default=600)
    parser.add_argument("--tb",                     help="number of timesteps with fitness avaliation, default is 5400", type=int, default=5400)
    parser.add_argument("-g", "--num-generations",  help="number of generations, default is 500", type=int, default=500)
    parser.add_argument("-r", "--num-runs",         help="number of runs, default is 3", type=int, default=3)
    parser.add_argument("-n", "--num-robots",       help="number of robots, default is 10", type=int, default=10)
    parser.add_argument("-p", "--population-size",  help="population size (genomes), default is 120", type=int, default=120)
    parser.add_argument("-d", "--distances",        help="list of distances between target areas to be evaluated each generation, default is 0.7 0.9 1.1 1.3 1.5", type=float, nargs='+', default=[0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("-t", "--trials",           help="number of trials per distance, default is 3", type=int, default=3)
    parser.add_argument("-c", "--pcrossover",       help="probability of crossover, default is 0.9", type=float, default=0.9)
    parser.add_argument("-m", "--pmutation",        help="prabability of mutation, default is 0.03", type=float, default=0.03)
    parser.add_argument("-e", "--elite-size",       help="size of population elite, default is 24", type=int, default=24)
    args = parser.parse_args()

    if args.verbosity >= 2:
        __log__.setLevel(logging.DEBUG)
    elif args.verbosity == 1:
        __log__.setLevel(logging.INFO)
    else:
        __log__.setLevel(logging.WARNING)

    if args.quiet:
        __log__.setLevel(logging.ERROR)

    uri = os.environ.get('SOLACE_URI')
    username = os.environ.get('SOLACE_USERNAME')
    password = os.environ.get('SOLACE_PASSWORD')

    if (uri is None) or (username is None) or (password is None):
        raise Exception('Environment variables (SOLACE_URI, SOLACE_USERNAME, SOLACE_PASSWORD) not set!')

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(args.num_runs, {
        'PCROSSOVER': args.pcrossover,
        'PMUTATION': args.pmutation,
        'ELITE_SIZE': args.elite_size,
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
        GA(context, queue).execute(run, args)

class GA(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def execute(self, run, args):
        __log__.info(' GA Starting...')
        __log__.info('=' * 80)

        run.begin()

        self.population = [ Individual() for i in range(args.population_size) ]
        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=args.population_size,
                                           num_robots=args.num_robots,
                                           ta=args.ta, tb=args.tb)

        last_best_fitness = 0

        __log__.info('Calculating initial fitness...')
        self.evaluate(self.population, args.distances)
        self.population = sorted(self.population, key=lambda ind: ind.fitness)

        generation = 1
        while (generation <= args.num_generations):
            genomeMom = None
            genomeDad = None

            new_pop = []
            elite = self.population[-args.elite_size:]

            size = len(self.population)
            if (size % 2) != 0:
                size -= 1

            for i in xrange(0, size, 2):
                genomeMom = self.select(self.population)
                genomeDad = self.select(self.population)

                if (args.pcrossover >= 1) or (random.random() < args.pcrossover):
                   (sister, brother) = genomeMom.crossover(genomeDad)
                else:
                    (sister, brother) = (genomeMom.copy(), genomeDad.copy())

                sister.mutate(args.pmutation)
                brother.mutate(args.pmutation)

                new_pop.append(sister)
                new_pop.append(brother)

            if len(self.population) % 2 != 0:
                last = self.select(self.population).copy()

                if (args.pmutation >= 1) or (random.random() < args.pmutation):
                    last.mutate()

                new_pop.append(last)

            __log__.info('[gen=%d] Evaluating population...', generation)
            self.evaluate(new_pop, args.distances)
            new_pop =  sorted(new_pop, key=lambda ind: ind.fitness)

            for i in xrange(args.elite_size):
                if elite[i].fitness > new_pop[i - args.elite_size].fitness:
                    new_pop[i - args.elite_size] = elite[i]

            self.population = sorted(new_pop, key=lambda ind: ind.fitness)

            best = self.population[-1]
            new_best = False
            if (best.fitness > last_best_fitness):
                new_best = True
                last_best_fitness = best.fitness

            __log__.info('[gen=%d] Population evaluated, current best individual: %s', generation, str(best))

            if new_best:
                run.progress(generation / float(args.num_generations), {
                    'generation': generation,
                    'best_fitness': best.fitness,
                    'best_genome': best.genome.to_dict()
                })

                if not args.no_save:
                    __log__.info('[gen=%d] Saving simulation for the new found best...', generation)
                    fit = self.simulate_and_save('/tmp/simulation.srs',
                            best.genome,
                            args.ta, args.tb,
                            args.num_robots,
                            args.distances[ random.randint(0, len(args.distances)-1) ])
                    run.upload('/tmp/simulation.srs', 'run-%02d-new-best-gen-%04d-fit-%.2f.srs' % (run.id, generation, fit) )
                    os.remove('/tmp/simulation.srs')
            else:
                run.progress(generation / float(args.num_generations), {'generation': generation})

            if args.image:
                self.generate_image('/tmp/image.png')
                run.upload('/tmp/image.png', 'image-run-%02d-gen-%04d.png' % (run.id, generation))
                os.remove('/tmp/image.png')

            generation += 1

        run.done({'generation': generation, 'best_fitness': best.fitness, 'best_genome': best.genome.to_dict()})

    def select(self, population):
        return population.pop()

    def evaluate(self, population, distances):
        for i in xrange(len(population)):
            params = population[i].genome
            self.simulator.set_ann_parameters(i, params)
        self.simulator.commit_ann_parameters()

        for i in xrange(len(population)):
            population[i].fitness = .0

        for d in distances:
            for i in range(3):
                self.simulator.init_worlds(d)
                self.simulator.simulate()

                fit = self.simulator.get_fitness()
                for i in xrange(len(population)):
                    population[i].fitness += fit[i]

        for i in xrange(len(population)):
            population[i].fitness /= len(distances) * 3

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
        save.add_object('arena', io.SHAPE_RECTANGLE, x=0.0, y=0.0, width=arena[0][0], height=arena[0][1])
        save.add_object('target0', io.SHAPE_CIRCLE, x=target_areas[0][0], y=target_areas[0][1], radius=target_areas_radius[0][0], sin=0.0, cos=0.1)
        save.add_object('target1', io.SHAPE_CIRCLE, x=target_areas[0][2], y=target_areas[0][3], radius=target_areas_radius[0][1], sin=0.0, cos=0.1)

        fitene = simulator.get_individual_fitness_energy()
        sensors, actuators, hidden = simulator.get_ann_state()
        transforms, radius = simulator.get_transforms()
        robot_radius = radius[0][0]

        robot_obj = [ None for i in range(len(transforms)) ]
        for i in range(len(transforms)):
            robot_obj[i] = save.add_object('robot'+str(i), io.SHAPE_CIRCLE,
                x=transforms[i][0], y=transforms[i][1], radius=robot_radius,
                sin=transforms[i][2], cos=transforms[i][3],
                fitness=fitene[i][0], energy=fitene[i][1],
                sensors=sensors[i],
                wheels0=actuators[i][0], wheels1=actuators[i][1],
                front_led=actuators[i][2], rear_led=actuators[i][3],
                hidden0=hidden[i][0], hidden1=hidden[i][1], hidden2=hidden[i][2])

        current_step = 0
        while current_step < (ta + tb):
            simulator.step()

            if (current_step <= ta):
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

            save.frame()
            current_step += 1

        save.close()

        return simulator.get_fitness()[0]

class Individual(object):
    def __init__(self):
        self.id = id(self)

        self.fitness = 0
        self.genome = physics.ANNParametersArray(True)

    def __repr__(self):
        return 'Individual(%d, fitness=%.5f)' % (self.id, self.fitness)

    def copy(self):
        g = Individual()
        g.fitness = self.fitness
        g.genome = self.genome.copy()
        return g

    def crossover(self, other):
       """ Single Point crossover """

       if len(self.genome) == 1:
          raise Exception('Cannot crossover genome of length 1.')

       point = random.randint(1, len(self.genome)-1)

       sister = self.copy()
       sister.genome.merge(point, other.genome)

       brother = other.copy()
       brother.genome.merge(point, self.genome)

       return (sister, brother)

    def mutate(self, pmutation):
        for i in xrange(len(self.genome.weights)):
            if random.random() < pmutation:
                self.genome.weights[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.bias)):
            if random.random() < pmutation:
                self.genome.bias[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.weights_hidden)):
            if random.random() < pmutation:
                self.genome.weights_hidden[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.bias_hidden)):
            if random.random() < pmutation:
                self.genome.bias_hidden[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.timec_hidden)):
            if random.random() < pmutation:
                self.genome.timec_hidden[i] += random.uniform(-1,1)

        self.genome.check_boundary(self.genome.weights_boundary, self.genome.weights)
        self.genome.check_boundary(self.genome.bias_boundary, self.genome.bias)
        self.genome.check_boundary(self.genome.weights_boundary, self.genome.weights_hidden)
        self.genome.check_boundary(self.genome.bias_boundary, self.genome.bias_hidden)
        self.genome.check_boundary(self.genome.timec_boundary, self.genome.timec_hidden)

if __name__=="__main__":
    main()