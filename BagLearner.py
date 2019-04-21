import numpy as np
import RTLearner
from random import randint


class BagLearner(object):

    def __init__(self,
                 learner,
                 kwargs,
                 bags,
                 boost=False,
                 verbose=False):

        self.learners = []
        for i in xrange(bags):
            self.learners.append(learner(**kwargs))

    @staticmethod
    def Get_Index(n):
        indices = []
        for i in xrange(int(n*0.6)):
            indices.append(randint(0, n-1))
        return indices

    def addEvidence(self, Xtrain, Ytrain):
        for learner in self.learners:
            indices = BagLearner.Get_Index(Xtrain.shape[0])
            
            Bag_Xtrain = []
            Bag_Ytrain = []
            for index in indices:
                Bag_Xtrain.append(Xtrain[index])
                Bag_Ytrain.append(Ytrain[index])
            learner.addEvidence(np.array(Bag_Xtrain), np.array(Bag_Ytrain))

    def query(self, Xtest):
        YtestResult = None
        for learner in self.learners:
            Bag_YtestResult = learner.query(Xtest)
            if YtestResult is None:
                YtestResult = Bag_YtestResult
            else:
                YtestResult = np.add(YtestResult, Bag_YtestResult)
                    #YtestResult /= len(self.learners)
        return YtestResult/len(self.learners)

    def author(self):
        return 'mmao33'
    
