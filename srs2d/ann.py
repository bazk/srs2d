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
__date__ = "03 Jul 2013"

import math
import random
import logging
import physics

__log__ = logging.getLogger(__name__)

NUM_INPUTS      = 13
NUM_OUTPUTS     = 4
NUM_HIDDEN      = 3

IN_camera0      =  0
IN_camera1      =  1
IN_camera2      =  2
IN_camera3      =  3
IN_proximity0   =  4
IN_proximity1   =  5
IN_proximity2   =  6
IN_proximity3   =  7
IN_proximity4   =  8
IN_proximity5   =  9
IN_proximity6   = 10
IN_proximity7   = 11
IN_ground0      = 12

OUT_wheels0     =  0
OUT_wheels1     =  1
OUT_front_led0  =  2
OUT_rear_led0   =  3

HID_hidden0     =  0
HID_hidden1     =  1
HID_hidden2     =  2

class NeuralNetworkController(object):
    def __init__(self):
        super(NeuralNetworkController, self).__init__()

        self.weights = [ [ random.uniform(-5.0, 5.0) for ih in range(NUM_INPUTS+NUM_HIDDEN) ] for o in range(NUM_OUTPUTS) ]
        self.bias = [ random.uniform(-5.0, 5.0) for o in range(NUM_OUTPUTS) ]

        self.weights_hidden = [ [ random.uniform(-5.0, 5.0) for i in range(NUM_INPUTS) ] for h in range(NUM_HIDDEN) ]
        self.bias_hidden = [ random.uniform(-5.0, 5.0) for h in range(NUM_HIDDEN) ]
        self.timec_hidden = [ random.uniform(0, 1.0) for h in range(NUM_HIDDEN) ]

        self.H = [ 0.0 for h in range(NUM_HIDDEN) ]

    def export(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'weights_hidden': self.weights_hidden,
            'bias_hidden': self.bias_hidden,
            'timec_hidden': self.timec_hidden
        }

    def load(self, data):
        self.weights = data['weights']
        self.bias = data['bias']
        self.weights_hidden = data['weights_hidden']
        self.bias_hidden = data['bias_hidden']
        self.timec_hidden = data['timec_hidden']

    def think(self, sensors, actuators):
        def sigmoid(z):
            return 1.0 / (1.0 + math.exp(-z))

        for h in range(NUM_HIDDEN):
            aux = 0.0
            for i in range(NUM_INPUTS):
                aux += self.weights_hidden[h][i] * \
                    sensors[i] + self.bias_hidden[h]

            self.H[h] = self.timec_hidden[h] * self.H[h] + \
                  (1 - self.timec_hidden[h]) * sigmoid(aux)

        for o in range(NUM_OUTPUTS):
            aux = 0.0
            for i in range(NUM_INPUTS):
                aux += self.weights[o][i] * sensors[i]
            for h in range(NUM_HIDDEN):
                aux += self.weights[o][h+NUM_INPUTS] * self.H[h]
            aux += self.bias[o]

            actuators[o] = sigmoid(aux)

        actuators[OUT_front_led0] = 1.0;
        actuators[OUT_rear_led0] = 1.0;