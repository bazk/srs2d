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

class NeuralNetworkController(physics.Controller):
    def __init__(self):
        super(NeuralNetworkController, self).__init__()

        self.connect('think', self.on_think)

        self._inputs = [
            'camera0',
            'camera1',
            'camera2',
            'camera3',
            'proximity3',
            'proximity2',
            'proximity1',
            'proximity0',
            'proximity7',
            'proximity6',
            'proximity5',
            'proximity4',
            'ground0'
        ]

        self._hiddens = [
            'hidden0',
            'hidden1',
            'hidden2'
        ]

        self._outputs = [
            'wheels1',
            'wheels0',
            'rear_led0',
            'front_led0'
        ]

        self.weights = { o: { ih: random.uniform(-5.0, 5.0) for ih in self._inputs + self._hiddens } for o in self._outputs }
        self.bias = { o: random.uniform(-5.0, 5.0) for o in self._outputs }

        self.weights_hidden = { h: { i: random.uniform(-5.0, 5.0) for i in self._inputs } for h in self._hiddens }
        self.bias_hidden = { h: random.uniform(-5.0, 5.0) for h in self._hiddens }
        self.timec_hidden = { h: random.uniform(0, 1.0) for h in self._hiddens }

        self.H = { h: 0.0 for h in self._hiddens }

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

    def on_think(self):
        self.update_sensors()

        def sigmoid(z):
            return 1.0 / (1.0 + math.exp(-z))

        for h in self._hiddens:
            aux = 0.0
            for i in self._inputs:
                aux += self.weights_hidden[h][i] * \
                    self.sensors[i] + self.bias_hidden[h]

            self.H[h] = self.timec_hidden[h] * self.H[h] + \
                  (1 - self.timec_hidden[h]) * sigmoid(aux)

        for o in self._outputs:
            aux = 0.0
            for i in self._inputs:
                aux += self.weights[o][i] * self.sensors[i]
            for h in self._hiddens:
                aux += self.weights[o][h] * self.H[h]
            aux += self.bias[o]

            self.actuators[o] = sigmoid(aux)

        self.update_actuators()