from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from math import exp
from random import random
from matplotlib import pyplot as plt
import pandas as pd





class NeuralNetworkClassifier:

    def __transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def __activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def __update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    # Forward propagate input to a network output
    def __forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.__activate(neuron['weights'], inputs)
                neuron['output'] = self.__transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def __transfer_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def __backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.__transfer_derivative(neuron['output'])

    def fit(self, data, target, n_of_hidden_nodes, n_of_outputs, l_rate, cycles):

        network = list()

        if len(data.shape) > 1:
            number_of_inputs = data.shape[1]
        else:
            number_of_inputs = 1

        hidden_layer = [{'weights': [random() for i in range(number_of_inputs + 1)]} for i in range(n_of_hidden_nodes)]
        output_layer = [{'weights': [random() for i in range(n_of_hidden_nodes + 1)]} for i in range(n_of_outputs)]

        network.append(hidden_layer)
        network.append(output_layer)

        graph = []
        for n in range(cycles - 1):
            accuracy = 0
            for x in range(len(data)):
                outputs = self.__forward_propagate(network, data[x])
                expected = [0 for i in range(n_of_outputs)]
                expected[target[x]] = 1
                if (np.argmax(expected) == np.argmax(outputs)):
                    accuracy += 1
                self.__backward_propagate_error(network, expected)
                self.__update_weights(network, data[x], l_rate)
            accuracy = accuracy / len(data)
            graph = np.append(graph, accuracy)
        return NeuralNetworkModel(network), graph

class NeuralNetworkModel:

    def __init__(self, network):
        self.network = network

    def __transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def __activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # Forward propagate input to a network output
    def __forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.__activate(neuron['weights'], inputs)
                neuron['output'] = self.__transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def predict(self, test):
        outputs_list = []

        for row in test:
            cur_output = self.__forward_propagate(self.network, row)
            cur_output =  np.argmax(self.__forward_propagate(self.network, row))
            outputs_list.append(cur_output)

        return outputs_list
		
		
# Iris dataset
iris = datasets.load_iris()

my_classifier_accuracy = 0
scikit_classifier_accuracy = 0
kf = KFold(n_splits=10, shuffle=True)

print("Iris dataset")
for train, test in kf.split(iris.data):
    data_train, data_test, target_train, target_test = iris.data[train], iris.data[test], iris.target[train], iris.target[test]

    classifier = NeuralNetworkClassifier()
    model, graph = classifier.fit(data_train, target_train, 4, 3, 0.08, 1000)
    targets = model.predict(data_test)

    corrects = 0

    for x in range(len(target_test)):
        if (target_test[x] == targets[x]):
            corrects += 1

    print("Accuracy: {}".format(corrects / len(target_test)))

    my_classifier_accuracy += corrects / len(target_test)
    plt.plot(graph)
    plt.ylabel('Accuracy')
    plt.xlabel('Loop')
    plt.title('Iris')
    plt.show()

    mlp = MLPClassifier(hidden_layer_sizes=(4), learning_rate_init=0.08, max_iter=1000)
    mlp.fit(data_train, target_train)
    predictions = mlp.predict(data_test)

    corrects = 0
    for x in range(len(target_test)):
        if (target_test[x] == predictions[x]):
            corrects += 1

    scikit_classifier_accuracy += corrects / len(target_test)

print("My accuracy: {}".format(my_classifier_accuracy / 10))
print("Scikits accuracy: {}".format(scikit_classifier_accuracy / 10))



# Pima Indian Diabetes
headers = ['times_pregnant', 'glucose', 'blood_pressure', 'triceps', 'insulin'
    , 'bmi', 'dpf', 'age', 'result']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', sep=",", header=None, names=headers)

# mark zero values as missing or NaN
data[['times_pregnant', 'glucose', 'blood_pressure', 'triceps', 'insulin']] = data[
    ['times_pregnant', 'glucose', 'blood_pressure', 'triceps', 'insulin']].replace(0, np.NaN)

# drop rows with missing values
data.dropna(inplace=True)

# Shuffling
data = data.sample(frac=1)

# Normalizing data
std_scale = preprocessing.StandardScaler().fit(data[['times_pregnant', 'glucose', 'blood_pressure', 'triceps', 'insulin'
    , 'bmi', 'dpf', 'age']])
data_std = std_scale.transform(data[['times_pregnant', 'glucose', 'blood_pressure', 'triceps', 'insulin'
    , 'bmi', 'dpf', 'age']])

# Getting target
target = np.array(data['result'])

scikit_classifier_accuracy = 0
my_classifier_accuracy = 0

print()
print("Pima Indian Diabetes")

# training / test sets
for train, test in kf.split(data_std):
    data_train, data_test, target_train, target_test = data_std[train], data_std[test], target[train], target[test]

    classifier = NeuralNetworkClassifier()
    model, graph = classifier.fit(data_train, target_train, 4, 2, 0.1, 1500)
    targets = model.predict(data_test)
    corrects = 0
    for x in range(len(target_test)):
        if (target_test[x] == targets[x]):
            corrects += 1

    print("Accuracy: {}".format(corrects / len(target_test)))
    my_classifier_accuracy += corrects / len(target_test)
    plt.plot(graph)
    plt.ylabel('Accuracy')
    plt.xlabel('Loop')
    plt.title('Pima Indian Diabetes')
    plt.show()

    mlp = MLPClassifier(hidden_layer_sizes=(4), learning_rate_init=0.1, max_iter=1500)
    mlp.fit(data_train, target_train)
    predictions = mlp.predict(data_test)
    corrects = 0
    for x in range(len(target_test)):
        if (target_test[x] == predictions[x]):
            corrects += 1

    scikit_classifier_accuracy += corrects / len(target_test)

print("My classifier accuracy: {}".format(my_classifier_accuracy / 10))
print("Scikit classifier accuracy: {}".format(scikit_classifier_accuracy / 10))