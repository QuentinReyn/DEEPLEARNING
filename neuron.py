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

    def meanSquareErrorLoss(self,expected,result):
        return np.square(np.subtract(expected, result)).mean()

    def gradient(self,x1,x2,expected,result):
        gradients = []
        nw1 = self.w1 - (x1 * ((2 * expected) - (2*result)))
        nw2 = self.w2 - (x2 * ((2 * expected) - (2*result)))
        gradients.append(nw1)
        gradients.append(nw2)
        return gradients

if __name__ == "__main__":
    myneuron = Neuron()
    x1 = 3
    x2 = 4
    x = myneuron.forward(x1, x2) #result
    expected = x1+x2
    print(myneuron.meanSquareErrorLoss(x1+x2,x))
    print(myneuron.gradient(x1,x2,expected,x))     
    # on a soit l'un soit l'autre, soit entre les 2
    # donc il prédit l'un ou l'autre ou il ne sait pas lequel c'est
    # print('x1 = {} et x2 = {}, res = {}'.format(x1, x2, myneuron.activation(x)))
