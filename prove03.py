import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing


# KNNeighborsClassifier
class HardCodedClassifierModel:

    def __init__(self, data, target):
        self.data_trained = np.array(data)
        self.target_trained = np.array(target)

    def __get_distances(self, test_ele):

        distances_array = []

        for x in self.data_trained:
            # Euclidean distance
            dist = np.linalg.norm(test_ele - x)
            # Adding distance to array
            distances_array = np.append(distances_array, dist)

        # Add axis to the distances
        distances_array = np.swapaxes(distances_array, 0, 1)
        distances_array = np.array([distances_array, self.target_trained])

        # Sorting array by distance
        distances_array = distances_array[distances_array[:, 0].argsort()]
        return distances_array


    def predict(self, test, k):

        target = []

        for ele in test:
            distances = self.__get_distances(ele)
            possible_targets = distances[:k, 1]
            values, counts = np.unique(possible_targets, return_counts=True)
            value_count = np.array([values, counts])
            value_count = np.swapaxes(value_count, 0, 1)
            value_count = value_count[value_count[:, 1].argsort()]
            target = np.append(target, value_count[::-1, 0][0])

        target = target.astype(int)
        return target


class KNNeighborsClassifier:

    def fit(self, data, target):
        return HardCodedClassifierModel(data, target)


headers = ['buying_price', 'maintenance_price', 'number_of_doors', 'capacity', 'luggage_boot_size'
    , 'safety', 'evaluation']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', sep=",", header=None, names=headers)

data["buying_price"] = data["buying_price"].astype('category')
data["maintenance_price"] = data["maintenance_price"].astype('category')
data["number_of_doors"] = data["number_of_doors"].astype('category')
data["capacity"] = data["capacity"].astype('category')
data["luggage_boot_size"] = data["luggage_boot_size"].astype('category')
data["safety"] = data["safety"].astype('category')
data["evaluation"] = data["evaluation"].astype('category')

data["buying_price_cat"] = data["buying_price"].cat.codes
data["maintenance_price_cat"] = data["maintenance_price"].cat.codes
data["number_of_doors_cat"] = data["number_of_doors"].cat.codes
data["capacity_cat"] = data["capacity"].cat.codes
data["luggage_boot_size_cat"] = data["luggage_boot_size"].cat.codes
data["safety_cat"] = data["safety"].cat.codes
data["evaluation_cat"] = data["evaluation"].cat.codes

data = data.sample(frac=1)

std_scale = preprocessing.StandardScaler().fit(data[['buying_price_cat', 'maintenance_price_cat', 'number_of_doors_cat',
                                                     'capacity_cat', 'luggage_boot_size_cat', 'safety_cat']])
data_std = std_scale.transform(data[['buying_price_cat', 'maintenance_price_cat', 'number_of_doors_cat', 'capacity_cat',
                                     'luggage_boot_size_cat', 'safety_cat']])
target = np.array(data['evaluation_cat'])

kf = KFold(n_splits=10)
for train, test in kf.split(data):
    data_train, data_test, target_train, target_test = data_std[train], data_std[test], target[train], target[test]

    # Using k = 17 (KNN)
    k = 17
    classifier = KNNeighborsClassifier()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test, k)

    corrects = 0

    for x in range(len(data_test)):
        if target_test[x] == targets_predicted[x]:
            corrects += 1

    print('Accuracy: {}/{} => {}'.format(corrects, len(test), corrects / len(test)))

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

kf = KFold(n_splits=10)
for train, test in kf.split(data):
    data_train, data_test, target_train, target_test = data_std[train], data_std[test], target[train], target[test]
    # Using k = 9 (KNN)
    k = 17
    classifier = KNNeighborsClassifier()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test, k)


    corrects = 0

    for x in range(len(data_test)):
        if target_test[x] == targets_predicted[x]:
            corrects += 1

    print('Accuracy: {}/{} => {}'.format(corrects, len(test), corrects / len(test)))

# Auto-mpg
headers = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight'
    , 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', delim_whitespace=True, header=None, names=headers)


# mark zero values as missing or NaN
data[['horsepower']] = data[['horsepower']].replace('?', np.NaN)
# drop rows with missing values
data.dropna(inplace=True)

# Shuffling
data = data.sample(frac=1)

# Handling categorical data
data["cylinders"] = data["cylinders"].astype('category')
data["model_year"] = data["model_year"].astype('category')
data["origin"] = data["origin"].astype('category')

data["cylinders_cat"] = data["cylinders"].cat.codes
data["model_year_cat"] = data["model_year"].cat.codes
data["origin_cat"] = data["origin"].cat.codes

# Normalizing data
std_scale = preprocessing.StandardScaler().fit(data[['cylinders_cat', 'displacement', 'horsepower', 'weight', 'acceleration'
    , 'model_year_cat', 'origin_cat']])
data_std = std_scale.transform(data[['cylinders_cat', 'displacement', 'horsepower', 'weight', 'acceleration'
    , 'model_year_cat', 'origin_cat']])

# Getting target
target = np.array(data['mpg'])

kf = KFold(n_splits=10)
for train, test in kf.split(data):
    data_train, data_test, target_train, target_test = data_std[train], data_std[test], target[train], target[test]
    # Using k = 9 (KNN)
    k = 17
    classifier = KNNeighborsClassifier()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test, k)

    corrects = 0

    for x in range(len(data_test)):
        if target_test[x] == targets_predicted[x]:
            corrects += 1

    print('Accuracy: {}/{} => {}'.format(corrects, len(test), corrects / len(test)))