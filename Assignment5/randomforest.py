from decision_tree import decision_tree
from random import seed, randrange, random

class randomforest(decision_tree):
    def __init__(self):
        self.n_folds = 5
        self.max_depth = 20
        self.min_size = 1
        self.sample_size = 1.0
    
    def fit(self, X, Y):
        self.build_tree(X, Y)

    def loadDataSet(self, filename):
        dataset = []
        with open(filename, 'r') as fr:
            for line in fr.readlines():
                if not line:
                    continue
                lineArr = []
                for featrue in line.split(','):
                    str_f = featrue.strip()
                    if str_f.isdigit():
                        lineArr.append(float(str_f))
                    else:
                        lineArr.append(str_f)
                dataset.append(lineArr)
        return dataset

    def cross_validation_split(self, dataset, n_folds):

        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = len(dataset) / n_folds
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy[index])
            dataset_split.append(fold)
        return dataset_split

    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def gini_index(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += (proportion * (1.0 - proportion))
        return gini


    def get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))  # class_values =[0, 1]
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
        for index in features:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
                
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del(node['groups'])

        if not left or not right:
            node['left'] = node['right'] = to_terminal(left + right)
            return

        if depth >= max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return

        if len(left) <= min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = get_split(left, n_features)
            split(node['left'], max_depth, min_size, n_features, depth+1)

        if len(right) <= min_size:
            node['right'] = to_terminal(right)
        else:
            node['right'] = get_split(right, n_features)
            split(node['right'], max_depth, min_size, n_features, depth+1)


    def build_tree(self, train, max_depth, min_size, n_features):

        root = get_split(train, n_features)

        split(root, max_depth, min_size, n_features, 1)
        return root

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']


    def bagging_predict(self, trees, row):

        predictions = [predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def subsample(self, dataset, ratio):

        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):

        trees = list()

        for i in range(n_trees):
            sample = subsample(train, sample_size)
            tree = build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)

        predictions = [bagging_predict(trees, row) for row in test]
        return predictions



    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
       
        folds = cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]

            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores