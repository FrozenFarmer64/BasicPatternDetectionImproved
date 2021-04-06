import random;
import numpy;
from numpy import array;

"""
BasicPatternDetectionImproved.py
Creator: Calvin Osborne

An adaptive, feedforward neural network with variable layer sizes and lengths.

Tested below with the following specifications:
input = [[x], [y]]
desired_output = (x > y) ? 1 : 0 
"""

# Compute the cost
# mode = 0: C
# mode = 1: dC/dy
def calculateCost(y, yH, mode):
    # Quadratic cost:
    if(False):
        if(mode == 0):
            ret = [];
            for i in range(0, y.shape[0]): ret.append([0.5 * ((y[i, 0] - yH[i, 0]) ** 2)])
            return array(ret);
        elif(mode == 1):
            dCdy = [];
            for i in range(y.shape[0]): dCdy.append([y[i, 0] - yH[i, 0]]);
            return dCdy;

    # Cross-entropy cost:
    # Note that this cost function is most effective when the desired outputs are 0 and 1
    if(True):
        if(mode == 0):
            ret = [];
            for i in range(0, y.shape[0]): ret.append([-1 * (yH[i, 0] * numpy.log(y[i, 0]) + (1 - yH[i, 0]) * numpy.log(1 - y[i, 0]))])
            return array(ret);
        elif(mode == 1):
            dCdy = [];
            for i in range(y.shape[0]): dCdy.append([(1 - yH[i, 0]) / (1 - y[i, 0]) - yH[i, 0] / y[i, 0]]);
            return dCdy;
    
    return None;

# The average function to be used at the end of the training method
def avgEnd(arrays):
    n = 1;
    ret = arrays[0];
    for i in range(1, len(arrays)):
        for j in range(0, len(ret)):
            ret[j] = ret[j] + arrays[i][j];
        n += 1;
    for i in range(0, len(ret)):
        ret[i] = ret[i] * (1 / n);
    return ret;

class NeuralNetwork:
    def __init__(self, numberInLayer, trainingRate):
        self.length = len(numberInLayer) - 1;
        self.weights = [];
        self.biases = [];
        for i in range(0, len(numberInLayer) - 1):
            # Initialize the weights for the layer
            values = [];
            for j in range(0, numberInLayer[i + 1]):
                temp = [];
                for k in range(0, numberInLayer[i]):
                    # Random value between 0 and -1
                    temp.append(random.random() * 2 - 1);
                values.append(temp);
            self.weights.append(array(values));
            # Initialize the biases for the layer
            values = [];
            for j in range(0, numberInLayer[i + 1]):
                values.append([random.random()]);
            self.biases.append(array(values));

        # The training rate will be multipied by the gradient
        self.trainingRate = trainingRate;

    def activation(self, x):
        return 1 / (1 + numpy.exp(-x));

    # The derivative of the inverse of the activation
    # o'(o^-1(y)) = o(o^-1(y))(1-o(o^-1(y))) = y(1 - y)!
    def dActivation(self, y):
        return y * (1 - y);

    def feedFoward(self, inputs):
        y = self.feedFowardLayers(inputs);
        return y[len(y) - 1];

    def feedFowardLayer(self, inputs, n):
        ret = numpy.dot(self.weights[n], inputs) + self.biases[n];
        for i in range(0, ret.shape[0]): ret[i, 0] = self.activation(ret[i, 0]);
        return ret;

    def feedFowardLayers(self, inputs):
        ret = [];
        ret.append(inputs);
        for i in range(0, self.length):
            temp = self.feedFowardLayer(inputs, i);
            ret.append(temp);
            inputs = temp;
        return ret;

    def gradient(self, inputs, answer):
        y = self.feedFowardLayers(inputs); # y (our outputs)
        yH = answer; # y^hat (the "answer")
        
        # Evaluate the cost
        C = calculateCost(y[self.length], yH, 0);

        # Set up the three important elements
        nablaBiases = [-1 for i in range(0, self.length)];
        nablaWeights = [-1 for i in range(0, self.length)];
        delta = calculateCost(y[self.length], yH, 1) * self.dActivation(y[self.length]); # delta^n = nablaCost/y^n * o'(z^n)

        # Run the back propogation process
        for i in range(self.length - 1, 0 - 1, -1):
            nablaBiases[i] = delta; # nabla b^l = delta^(l + 1)
            nablaWeights[i] = numpy.dot(delta, y[i].transpose()); # nabla w^l = delta^(l + 1) (y^l)^T
            delta = numpy.dot(self.weights[i].transpose(), delta) * self.dActivation(y[i]); # delta^l = ((w^l)^T delta^(l + 1)) * o'(z^l)
            
        return [nablaWeights, nablaBiases, [C]];

    def train(self, inputs, answers):
        weightsGradient = [];
        biasesGradient = [];
        costArray = [];

        # Run through every test case
        for i in range(0, len(inputs)):
            gradient = self.gradient(inputs[i], answers[i]);
            weightsGradient.append(gradient[0]);
            biasesGradient.append(gradient[1]);
            costArray.append(gradient[2]);

        # Average the results
        weightsGradient = avgEnd(weightsGradient);
        biasesGradient = avgEnd(biasesGradient);
        costArray = avgEnd(costArray);
        
        # Subtract the results
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + weightsGradient[i] * -self.trainingRate;
            self.biases[i] = self.biases[i] + biasesGradient[i] * -self.trainingRate;

        return costArray[0];

    def print(self):
        print("Weights:")
        for layer in network.weights:
            print(layer);
        print("\nBiases:");
        for layer in network.biases:
            print(layer);
    
# Set the seed
random.seed(0);

# Set the parameters
training_size = 100;
training_iterations = 1000;

# Setup the training data
trainingData = [];
trainingAnswers = [];
for i in range(0, training_size): trainingData.append(array([[random.random()], [random.random()]]));
for i in range(0, training_size): trainingAnswers.append(array([[int(trainingData[i][0, 0] > trainingData[i][1,0])]]));

# Setup the network
network = NeuralNetwork([2, 2, 1], 0.5);

print("Training Data:");
for i in range(0, training_size): print("Trial " + str(trainingData[i]) + ": " + str(network.feedFoward(trainingData[i])));

print("\nTraining:");
for i in range(0, training_iterations):
    cost = network.train(trainingData, trainingAnswers);
    if(i % 1 == 0): print("Training " + str(i) + " complete: C = " + str(cost));
    
print("\nResults:");
for i in range(0, training_size): print("Trial " + str(trainingData[i]) + ": " + str(network.feedFoward(trainingData[i])));
