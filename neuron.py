#!/usr/bin/python3
# -*-coding:Utf-8 -*
import random as r
import numpy as np


class Neuron():

    def __init__(self):
        np.random.seed(10)
        # r.seed(10)
        self.w1 = r.random()
        self.w2 = r.random()

    # prend entrées, qui peuvent etre une photo de chat
    def forward(self, x1, x2):
        return (x1*self.w1 + x2*self.w2)

    def activation(self, input):
        return input

    ###### Tools ######

    def getRandomVector(self, size):
        return np.random.rand(size)

    def getRandomMatrice(self, width, height):
        return np.random.rand(height, width)

    def getSumResult(self, inputMat):
        matResult = []
        for x in inputMat:
            matResult.append(sum(x))
        return matResult

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_inputs, training_outputs, training_iterations):

        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            # computing error rate for back-propagation
            error = training_outputs - output

            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error *
                                 self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments


if __name__ == "__main__":
    myneuron = Neuron()
    # on entre 10 photos de chats
    x1 = 4
    x2 = 40
    x = myneuron.forward(x1, x2)
    # on a soit l'un soit l'autre, soit entre les 2
    # donc il prédit l'un ou l'autre ou il ne sait pas lequel c'est
    print('x1 = {} et x2 = {}, res = {}'.format(x1, x2, myneuron.activation(x)))
