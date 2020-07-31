import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(10)
samples = 20 #number of training, and also test samples
training_inputs = []  # Training data vector
test_inputs = []  # Test data vector
labels = []  # Supervised Training, this vector will include the tags for the training

class Perceptron():
    def __init__(self, no_inputs = 2, max_iter = 1000):#,learning_rate=0.01):
        self.max_iter = max_iter #maximum number of iterations
        self.weights = np.zeros(no_inputs + 1) #as b = w0, so w0 must be included in W
    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights [0] #dot product between x and w
        if sum > 0: #sign function with {-1; +1} outputs
            value = 1
        else:
            value = -1
        return value
    def train(self, training_inputs, labels):
        trained = False #used to control the training cycles
        iterations = 0  #count the number of iterations
        while not trained:
            error_count = 0.00 #control variable
            for inputs, label in zip (training_inputs, labels): # [X, tags], Tags: {-1,+1}
                prediction = self.predict(inputs)
                if label != prediction: #if the estimation W is different than the actual m
                    error = label - prediction #we feed forward with the error
                    self.weights[1:] += error * inputs # learning rate asumed to be 1.
                    self.weights[0:] += error #different dimension
                    error_count += abs(error) #increase the error coutner
            iterations += 1 #increase number of iterations
            if error_count == 0.00 or iterations >= self.max_iter:
                 print('# of Iterations: {}'.format(iterations)) #print max number of iterations
                 trained = True
def F_x (x): #the choosen random line in the plane, a simple form y = x
    y = x
    return y

def Data_generation(samples):
    for i in range (samples):
        x_0= random.randint(0,samples) #radon value for x_axis in training
        x_1= random.randint(0,samples)#random value for y_axis and calibration in training
        training_inputs.append(np.array([x_0, x_1])) #training inputs array with thw previous values

        f= F_x(x_0)#the reference, used to label the data with the Over/Under the line criteria
        if x_1 >= f:  #if the test is over the reference line
            labels.append(1) #Over the line
        else:
             labels.append(-1)#Under the line
    return training_inputs, labels

#INIT AND TESTING
Data_generation(samples)
perceptron = Perceptron() #initiates the perceptron
perceptron.train(training_inputs, labels) # initiates training
weights = perceptron.weights

#GRAPHICS
i = 0
while i < len(training_inputs):
    #graph for the test points: +/_ markings
    if ((perceptron.predict([training_inputs[i][0], training_inputs[i][1]])) == 1):
        c = "blue"
        m  ="+"
    else:
        c = "red"
        m = "_"
    plt.scatter(training_inputs[i][0], training_inputs[i][1], color= c, marker=m, s=50)

    #graph of the hypothesis.
    slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
    intercept = -weights[0] / weights[2]
    y = (slope * i) + intercept # y =mx+b, m is slope and b is intercept
    i += 1
    plt.plot(i, y, 'k^', label = "g(x)"), plt.plot(i, F_x(i), 'go', label = "f(x)")

plt.grid(False), plt.ylabel("y-axis"), plt.xlabel("x-axis"), plt.title("Perceptron algorithm")
plt.show()