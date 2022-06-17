from sklearn.datasets import load_iris
import numpy as np

class NBC:
    # define the constructor to create NBC object
    def __init__(self, feature_types=None, num_classes=None):
        # check constructor parameters
        if feature_types is None:
            self.feature_types = ['r', 'r', 'r', 'r']
        else:
            self.feature_types = feature_types

        if num_classes is None:
            self.num_classes = 3
        else:
            self.num_classes = num_classes

    # load the iris dataset
    def loadDataset(self):
        iris = load_iris()
        X, y = iris['data'], iris['target']
        return X, y

    # split and shuffle the dataset
    def splitAndShuffle(self, X, y):
        N, D = X.shape
        Ntrain = int(0.8 * N)
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]
        return Xtrain, ytrain, Xtest, ytest

    # estimate all the parameters of the NBC
    def fit(self, X, y):
        # count the number of elements in each class
        classtypes, counts = np.unique(y, return_counts=True)

        # calculate the prior distribution / class probability
        priors = np.array([float(count / sum(counts)) for count in counts])

        # separate the training data by class
        X_by_class = [X[y == classtype] for classtype in classtypes]

        # compute the conditional distributions
        # the mean and std of each feature in each class
        means = np.array([X_by_class[classtype].mean(axis=0) for classtype in classtypes])
        stds = np.array([X_by_class[classtype].std(axis=0) for classtype in classtypes])

        # set variance/std to small number 1e-6
        stds[stds == 0] = 10 ** -6

        # set the private variables for the use of other class methods
        self._classes = np.unique(y)
        self._priors = priors
        self._means = means
        self._stds = stds

        return self

    # compute gaussian distribution
    def gaussDistribution(self, x, mean, std):
        gauss_dis = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return gauss_dis

    # predict the class of each testing point that has the largest probability among the classes
    def predictProb(self, x):
        posteriors = []

        # iterate over classes
        for i, c in enumerate(self._classes):
            # pick the prior class prob
            prior = np.log(self._priors[i])

            # compute the feature prob of four features
            feature_prob = np.log(self.gaussDistribution(x, self._means[i], self._stds[i]))

            # sum the feature distributions condition on class in log space
            cond = np.sum(feature_prob)

            # compute the posterior class prob
            posterior = prior + cond

            # append the posterior class prob to the list
            posteriors.append(posterior)

        # return the class that has the largest probability
        return self._classes[np.argmax(posteriors)]

    # predict the classes in testing dataset
    def predict(self, Xtest):
        yPred = []

        # iterate over testing points
        for x in Xtest:
            # append the class in the prediction set
            yPred.append(self.predictProb(x))
        return yPred


def main():
    # instantiate a NBC instance
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)

    # load the data set: X is features vector, Y is list of label
    X, y = nbc.loadDataset()

    # split and suffle the data set
    Xtrain, ytrain, Xtest, ytest = nbc.splitAndShuffle(X, y)

    # train the model
    nbc.fit(Xtrain, ytrain)

    # predict the class
    yhat = nbc.predict(Xtest)

    # compute the model accuracy
    test_accuracy = np.mean(yhat == ytest)
    print(f"Testing accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
