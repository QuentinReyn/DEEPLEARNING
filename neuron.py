#!/usr/bin/python3
# -*-coding:Utf-8 -*
import random as r
import numpy as np


class Neuron():

    def __init__(self):
        np.random.seed(10)
        # r.seed(10)
        self.w1 = 0.5
        self.w2 = 0.5

    # prend entrées, qui peuvent etre une photo de chat
    def forward(self, inputs):
        return (inputs[0]*self.w1 + inputs[1]*self.w2)

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

    def meanSquareErrorLoss(self, inputs, result):
        expected = sum(inputs)
        return np.square(np.subtract(expected, result)).mean()

    def backPropagate(self, gradients):
        self.w1 = gradients[0]
        self.w2 = gradients[1]
        print('new w1', self.w1)
        print('new w2', self.w2)

class Trainer():

    def __init__(self,model):
        self.model = model
    
    def meanSquareErrorLoss(self, inputs, result):
        expected = sum(inputs)
        return np.square(np.subtract(expected, result)).mean()
    
    def gradient(self, inputs, result):
        expected = sum(inputs)
        gradients = []
        nw1 = 0
        nw2 = 0
        for i in range(len(inputs)):
            if i == 0:
                nw1 = self.model.w1 - (inputs[i] * ((2 * expected) - (2*result)))
            if i == 1:
                nw2 = self.model.w2 - (inputs[i] * ((2 * expected) - (2*result)))

        gradients.append(nw1)
        gradients.append(nw2)
        return gradients
    
    def train(self,inputs):
        for i in inputs:
            print(i)
            pred = self.model.forward(i)
            grad = self.gradient(i,pred)
            self.model.backPropagate(grad)
            print('Error loss:', myneuron.meanSquareErrorLoss(i, pred))          


if __name__ == "__main__":
    myneuron = Neuron()
    #inputs = [[3, 4],[3,4]]
    inputs = myneuron.getRandomMatrice(2,4)
    print(inputs)
    x1 = 3
    x2 = 4
    trainer = Trainer(myneuron)
    trainer.train(inputs)
