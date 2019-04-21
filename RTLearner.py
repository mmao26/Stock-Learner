import numpy as np
from random import randint


class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.RTDecTree = np.array([])

    def Get_FeatureIndex(self, x_train, num_samples):
        feature_i = randint(0, x_train.shape[1] - 1)
        RamData1 = randint(0, num_samples - 1)
        RamData2 = randint(0, num_samples - 1)
        SplitVal = (x_train[RamData1, feature_i] + x_train[RamData2, feature_i])/2
        lefttree = [i for i in range(x_train.shape[0]) if x_train[i, feature_i] <= SplitVal]
        righttree = [i for i in range(x_train.shape[0]) if x_train[i, feature_i] > SplitVal]
        return feature_i, SplitVal, lefttree, righttree

    def build_tree(self, x_train, y_train):
        if len(np.unique(y_train)) == 1:
            return np.array([-1, y_train[0], -1, -1])
        if x_train.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(y_train), -1, -1])

        feature_i, SplitVal, lefttree, righttree = self.Get_FeatureIndex(x_train, x_train.shape[0])

        while len(lefttree) < 1 or len(righttree) < 1:
            feature_i, SplitVal, lefttree, righttree = self.Get_FeatureIndex(x_train, x_train.shape[0])
        
        left_tree = self.build_tree(np.array([x_train[i] for i in lefttree]), np.array([y_train[i] for i in lefttree]))
        right_tree = self.build_tree(np.array([x_train[i] for i in righttree]), np.array([y_train[i] for i in righttree]))

        if len(left_tree.shape) == 1:
            num_LeftSample = 2
        else:
            num_LeftSample = left_tree.shape[0] + 1
        root_node = [feature_i, SplitVal, 1, num_LeftSample]
        return np.vstack((root_node, np.vstack((left_tree, right_tree))))

    def addEvidence(self, Xtrain, Ytrain):
        self.RTDecTree = self.build_tree(Xtrain, Ytrain)

    def Predict_tree(self, sample, row=0):
        feature_i = int(self.RTDecTree[row, 0])
        if feature_i == -1:
            return self.RTDecTree[row, 1]
        if sample[feature_i] <= self.RTDecTree[row, 1]:
            return self.Predict_tree(sample, row + int(self.RTDecTree[row, 2]))
        else:
            return self.Predict_tree(sample, row + int(self.RTDecTree[row, 3]))

    def query(self, Xtest):
        Ypredict = []
        for sample in Xtest:
            Ypredict.append(self.Predict_tree(sample))
        return np.array(Ypredict)

    def author(self):
        return 'mmao33'
