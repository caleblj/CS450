import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Step 1

iris = datasets.load_iris()

print(iris.data)

print(iris.target)

print(iris.target_names)

# Step 2

columns = "Sepal_length Sepal_width Petal_length Petal_width".split()
df = pd.DataFrame(iris.data, columns=columns)
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# Step 3

classifier = GaussianNB()
model = classifier.fit(x_train, y_train)

# Step 4
print(x_test)
targets_predicted = model.predict(x_test)

print(targets_predicted)
print("Percentage of sklearn corect:")
print(100 * model.score(x_test, y_test))


# Step 5

class HardCodedClassifier:

    def __init__(self):
        pass

    def fit(self, data, target):
        return HardCodedClassifier()

    def predict(self, test_data):
        target = []
        for x in test_data:
            target.append(0)
        return target

    def score(self, x_test, y_test):
        total = 0
        correct = 0
        for x in x_test:
            total += 1

        i = 0
        while i < total:
            if x_test[i] == y_test[i]:
                correct += 1
            i += 1

        return float(correct / total)

BlankClassifier = HardCodedClassifier()

BlankModel = BlankClassifier.fit(x_train, y_train)
BlankPredicted = BlankModel.predict(x_test)
print("Percentage of HardCoded correct:")
print(100 * BlankModel.score(BlankPredicted, y_test))