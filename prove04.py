import numpy as np
import pandas as pd
from anytree import Node, RenderTree, LevelOrderIter, findall
from sklearn import datasets
from sklearn.model_selection import KFold
import scipy  as sc


class DecisionTreeModel:
    def __get_probs(self, values):
        probs_array = []
        index = 0
        num_elements = len(values)

        for j in np.unique(values):
            for i in values:
                if i == j:
                    index += 1
            probs_array = np.append(probs_array, index / num_elements)
            index = 0
        return probs_array

    def __get_entropy(self, column, dataframe):
        grouped = dataframe.groupby(column)
        values_in_column = len(np.unique(dataframe[column]))
        total_entropy = 0
        for name, group in grouped:
            values_for_group = len(group['class'])
            p_data = group['class'].value_counts() / len(group)
            total_entropy += sc.stats.entropy(p_data, base=2) * values_for_group
        return total_entropy / values_in_column

    def __init__(self, data):
        self.data_trained = data

    def build_tree(self, node, dataframe):
        if (node.name == 'root'):
            entropy_dic = {}
            for column in self.data_trained:
                if column != 'class':
                    entropy = self.__get_entropy(column, self.data_trained)
                    entropy_dic[column] = entropy
            min_entropy = min(entropy_dic, key=entropy_dic.get)
            for x in np.unique(self.data_trained[min_entropy]):
                child = Node(min_entropy, value=x)
                child.parent = node
            grouped = dataframe.groupby(min_entropy)
            for child_node in LevelOrderIter(node, filter_=lambda n: n.name != node.name, maxlevel=2):
                dataframe_for_child = grouped.get_group(child_node.value)
                self.build_tree(child_node, dataframe_for_child)
        else:
            entropy_dic = {}
            for column in dataframe:
                if column != 'class':
                    entropy = self.__get_entropy(column, dataframe)
                    entropy_dic[column] = entropy
            min_entropy = min(entropy_dic, key=entropy_dic.get)
            if (entropy_dic[min_entropy] != 0):
                for x in np.unique(self.data_trained[min_entropy]):
                    child = Node(min_entropy, value=x)
                    child.parent = node
                grouped = dataframe.groupby(min_entropy)
                for child_node in LevelOrderIter(node, filter_=lambda n: n.name != node.name, maxlevel=2):
                    dataframe_for_child = grouped.get_group(child_node.value)
                    self.build_tree(child_node, dataframe_for_child)
            else:
                entropy_zero = True
                for key in entropy_dic:
                    entropy_zero = entropy_zero and (entropy_dic[key] == 0)
                if entropy_zero == True:
                    class_values = dataframe['class']
                    child = Node('class', value=class_values.iloc[0])
                    child.parent = node
                else:
                    entropy_dic = {k: v for k, v in entropy_dic.items() if k != 0}
                    min_entropy = min(entropy_dic, key=entropy_dic.get)
                    for x in np.unique(self.data_trained[min_entropy]):
                        child = Node(min_entropy, value=x)
                        child.parent = node
                    grouped = dataframe.groupby(min_entropy)
                    for child_node in LevelOrderIter(node, filter_=lambda n: n.name != node.name, maxlevel=2):
                        dataframe_for_child = grouped.get_group(child_node.value)
                        self.build_tree(child_node, dataframe_for_child)

    def predict(self, test):
        node = Node('root', value=0)
        self.build_tree(node, self.data_trained)
        for pre, fill, node_c in RenderTree(node):
            print("%s%s: %s" % (pre, node_c.name, node_c.value))
        targets_predicted = []

        for x in range(len(test)):
            temp_node = node
            while (temp_node.name != 'class'):
                for it_node in temp_node.children:
                    for col in test.columns:
                        if (it_node.name == col and it_node.value == test[col].iloc[x]):
                            temp_node = it_node
            targets_predicted = np.append(targets_predicted, temp_node.value)
        return targets_predicted

class DecisionTree:
    def fit(self, data):
        return DecisionTreeModel(data)

headers = ['index', 'age', 'spectacle', 'astigmatic', 'tear_production'
    , 'class']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data', delim_whitespace=True, header=None, names=headers)
target = data['class']

del data['index']

kf = KFold(n_splits=8, shuffle=True)
for train, test in kf.split(data):
    data_train, data_test, target_train, target_test = data.iloc[train], data.iloc[test], target[train], target[test]
    classifier = DecisionTree()
    model = classifier.fit(data)
    targets_predicted = model.predict(data_test)
    corrects = 0
    for x in range(len(data_test)):
        if target_test.iloc[x] == targets_predicted[x]:
            corrects += 1

    print('Accuracy: {}/{} => {:4.2f}'.format(corrects, len(test), corrects / len(test)))