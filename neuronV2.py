import random as r
import numpy as np
import matplotlib.pyplot as plt

r.seed(10)

def getRandomVector(size):
    return [r.random() for _ in range(size)]

def getRandomMatrice(width, height):
    return [getRandomVector(width) for _ in range(height)]

def getSumResult(inputMat):
    matResult = []
    for x in inputMat:
        matResult.append(sum(x))
    return matResult
class Neuron():
    def __init__(self):
        self.w1 = 0.5
        self.w2 = 0.5

    def activation(self,input):
        return input

    # prend entrÃ©es, qui peuvent etre une photo de chat
    def forward(self,inp):
        return self.activation(self.w1*inp[0] + self.w2*inp[1])
    
    def getSumInput(self, inp):
        return self.activation(inp[0] + inp[1])

    def getLoss(self, exp, res):
        """Calculate the loss with expected result and effective result"""
        return (exp - res)**2

    # nouveau poids
    def getGradient(self, inp, exp, res, lrate):
        return inp*(2.0*(exp-res))*lrate
    
    def backPropagate(self, grad):
        # met a jour les poids du neurone avec les deltas (grad)
        self.w1 -= grad[0]
        self.w2 -= grad[1]


myneuron = Neuron()
moninput = [3,4]
gradients = []
lrate = 0.00001
result = myneuron.forward(moninput)
expected = myneuron.getSumInput(moninput)
loss = myneuron.getLoss(expected, result)
errors = []
for j in range(1000):
    moninput = getRandomVector(2)
    result = myneuron.forward(moninput)
    expected = myneuron.getSumInput(moninput)
    for i in moninput:
        gradients.append(myneuron.getGradient(i, expected, result, lrate))
    print(f"{gradients}")
    error = myneuron.getLoss(expected, result)
    errors.append(error)
    print(f"{error}")
    myneuron.backPropagate(gradients)
    gradients = []

print(myneuron.forward([1,1]))
plt.plot(errors)
plt.xlabel("Error for all training instances")
plt.ylabel("Iterations")
plt.savefig("cumulative_error.png")
