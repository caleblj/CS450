from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()


# Step 2
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.2)


# Step 3
classifier = KNeighborsClassifier(n_neighbors = 3)
model = classifier.fit(x_train, y_train)

# Step 4
targets_predicted = model.predict(x_test)

print("Percentage of sklearn corect:")
print(100 * model.score(x_test, y_test))

# Step 5
class knnClassifier:

    k = 1
    data = []
    target = []

    def __init__(self, k, data=[], target=[] ):
        self.k = k
        self.data = data
        self.target = target

    def fit(self, data, target):
        self.data = data
        self.target = target
        return knnClassifier(self.k, self.data, self.target)

    def predict(self, test_data):
        nInputs = np.shape(test_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            #compute distances
            distances = np.sum((self.data-test_data[n, :])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(self.target[indices[:self.k]])

            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = np.max(counts)


        return closest


    def score(self, x_test, y_test):
        total = len(x_test)
        correct = 0

        for i in range(total):
            if x_test[i] == y_test[i]:
                correct += 1

        return float(correct) / total

BlankClassifier = knnClassifier(4)

BlankModel = BlankClassifier.fit(x_train, y_train)
BlankPredicted = BlankModel.predict(x_test)
print("Percentage of HardCoded correct:")
print(100 * BlankModel.score(BlankPredicted, y_test))