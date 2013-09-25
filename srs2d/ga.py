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

NUM_SENSORS     = 13
NUM_ACTUATORS   = 4
NUM_HIDDEN      = 3

STEPS_TA = 18600
STEPS_TB = 5400

NUM_GENERATIONS = 500
NUM_RUNS = 5
NUM_ROBOTS = 10
POPULATION_SIZE = 120

D = [0.9, 1.1, 1.3]

PCROSSOVER = 0.9
PMUTATION = 0.02
ELITE_SIZE = 20

class GA(object):
    def __init__(self, context, queue):
        self.context = context
        self.queue = queue

    def select(self):
        return self.population.pop()

    def evaluate(self):
        for i in xrange(len(self.population)):
            params = self.population[i].genome
            self.simulator.set_ann_parameters(i, params)
        self.simulator.commit_ann_parameters()

        for i in xrange(len(self.population)):
            self.population[i].fitness = .0

        for d in D:
            for i in range(3):
                self.simulator.init_worlds(d)
                self.simulator.simulate()

                fit = self.simulator.get_fitness()
                for i in xrange(len(self.population)):
                    self.population[i].fitness += fit[i]

        for i in xrange(len(self.population)):
            self.population[i].fitness /= len(D) * 3

    def execute(self, run):
        __log__.info(' GA Starting...')
        __log__.info('=' * 80)

        run.begin()

        self.population = [ Individual() for i in range(POPULATION_SIZE) ]
        self.simulator = physics.Simulator(self.context, self.queue,
                                           num_worlds=POPULATION_SIZE,
                                           num_robots=NUM_ROBOTS,
                                           ta=STEPS_TA, tb=STEPS_TB)

        last_best_fitness = 0

        __log__.info('Calculating initial fitness...')
        self.evaluate()
        self.population = sorted(self.population, key=lambda ind: ind.fitness)

        generation = 0
        while (generation < NUM_GENERATIONS):
            genomeMom = None
            genomeDad = None

            new_pop = []
            elite = self.population[-ELITE_SIZE:]

            size = len(self.population)
            if (size % 2) != 0:
                size -= 1

            for i in xrange(0, size, 2):
                genomeMom = self.select()
                genomeDad = self.select()

                if (PCROSSOVER >= 1) or (random.random() < PCROSSOVER):
                   (sister, brother) = genomeMom.crossover(genomeDad)
                else:
                    (sister, brother) = (genomeMom.copy(), genomeDad.copy())

                sister.mutate(PMUTATION)
                brother.mutate(PMUTATION)

                new_pop.append(sister)
                new_pop.append(brother)

            if len(self.population) % 2 != 0:
                last = self.select().copy()

                if (PMUTATION >= 1) or (random.random() < PMUTATION):
                    last.mutate()

                new_pop.append(last)

            __log__.info('Calculating fitness for each individual...')
            self.evaluate()
            new_pop =  sorted(new_pop, key=lambda ind: ind.fitness)

            __log__.info('-' * 80)
            __log__.info('ELITE')
            for i in xrange(ELITE_SIZE):
                __log__.info(str(elite[i]))
            __log__.info('-' * 80)
            __log__.info('NEW POP')
            for i in xrange(len(new_pop)):
                __log__.info(str(new_pop[i]))
            __log__.info('-' * 80)

            for i in xrange(ELITE_SIZE):
                if elite[i].fitness > new_pop[i - ELITE_SIZE].fitness:
                    new_pop[i - ELITE_SIZE] = elite[i]

            self.population = sorted(new_pop, key=lambda ind: ind.fitness)

            generation += 1

            best = self.population[-1]
            new_best = False
            if (best.fitness > last_best_fitness):
                __log__.info('Found new best individual: %s', str(best))
                new_best = True
                last_best_fitness = best.fitness

            __log__.info('-' * 80)
            __log__.info('[gen=%d] CURRENT BEST INDIVIDUAL IS: %s', generation, str(best))
            __log__.info(str(best.genome))
            __log__.info('-' * 80)

            run.progress(generation / float(NUM_GENERATIONS), {'generation': generation, 'best_fitness': best.fitness, 'best_genome': best.genome.to_dict()})

            if new_best:
                __log__.info('Saving simulation for the new found best...')
                fit = self.simulate_and_save('/tmp/simulation.srs', best.genome, D[ random.randint(0, len(D)-1) ])
                run.upload('/tmp/simulation.srs', 'run-%02d-new-best-gen-%04d-fit-%.2f.srs' % (run.id, generation, fit) )

        run.done({'generation': generation, 'best_fitness': best.fitness, 'best_genome': best.genome.to_dict()})

    def simulate_and_save(self, filename, pos, distance):
        simulator = physics.Simulator(self.context, self.queue,
                                      num_worlds=1,
                                      num_robots=NUM_ROBOTS,
                                      ta=STEPS_TA, tb=STEPS_TB)

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
        while current_step < (STEPS_TA + STEPS_TB):
            simulator.step()

            if (current_step <= STEPS_TA):
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

class Individual(object):
    def __init__(self):
        self.id = id(self)

        self.fitness = 0
        self.genome = physics.ANNParametersArray(True)

    def __str__(self):
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

    def mutate(self, pMutation):
        for i in xrange(len(self.genome.weights)):
            if random.random() < pMutation:
                self.genome.weights[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.bias)):
            if random.random() < pMutation:
                self.genome.bias[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.weights_hidden)):
            if random.random() < pMutation:
                self.genome.weights_hidden[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.bias_hidden)):
            if random.random() < pMutation:
                self.genome.bias_hidden[i] += random.uniform(-5,5)

        for i in xrange(len(self.genome.timec_hidden)):
            if random.random() < pMutation:
                self.genome.timec_hidden[i] += random.uniform(-1,1)

        self.genome.check_boundary(self.genome.weights_boundary, self.genome.weights)
        self.genome.check_boundary(self.genome.bias_boundary, self.genome.bias)
        self.genome.check_boundary(self.genome.weights_boundary, self.genome.weights_hidden)
        self.genome.check_boundary(self.genome.bias_boundary, self.genome.bias_hidden)
        self.genome.check_boundary(self.genome.timec_boundary, self.genome.timec_hidden)

if __name__=="__main__":
    uri = os.environ.get('SOLACE_URI')
    username = os.environ.get('SOLACE_USERNAME')
    password = os.environ.get('SOLACE_PASSWORD')

    if (uri is None) or (username is None) or (password is None):
        raise Exception('Environment variables (SOLACE_URI, SOLACE_USERNAME, SOLACE_PASSWORD) not set!')

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    exp = solace.get_experiment(uri, username, password)
    inst = exp.create_instance(NUM_RUNS, {
        'NUM_SENSORS': NUM_SENSORS,
        'NUM_ACTUATORS': NUM_ACTUATORS,
        'NUM_HIDDEN': NUM_HIDDEN,
        'PCROSSOVER': PCROSSOVER,
        'PMUTATION': PMUTATION,
        'ELITE_SIZE': ELITE_SIZE,
        'STEPS_TA': STEPS_TA,
        'STEPS_TB': STEPS_TB,
        'NUM_GENERATIONS': NUM_GENERATIONS,
        'NUM_RUNS': NUM_RUNS,
        'NUM_ROBOTS': NUM_ROBOTS,
        'POPULATION_SIZE': POPULATION_SIZE,
        'D': D
    })

    for run in inst.runs:
        GA(context, queue).execute(run)
