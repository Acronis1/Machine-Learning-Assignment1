import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import itertools
import pandas as pd
import os


class BOVHelper:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters, n_jobs=4)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC()
        self.clf2 = MLPClassifier()

    def cluster(self):
        """
        cluster using KMeans algorithm,
        """
        print("clustering")
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self,n_images, descriptor_list, kmeans_ret = None):

        """
        Each cluster denotes a particular visual word
        Every image can be represented as a combination of multiple
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word
        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images
        """

        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            print("develop vocabulary "+str(i)+"/"+str(n_images))
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count+j]
                else:
                    idx = kmeans_ret[old_count+j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print("Vocabulary Histogram Generated")
        self.plotHist()

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.
        """
        print("standardize")
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
        restructures list into vstack array of shape
        M samples x N features for sklearn
        """
        print("format ND")
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return vStack

    def train(self, train_labels, args):
        """
        uses sklearn.svm.SVC classifier (SVM)
        """
        print("Training SVM")
        print(self.clf)
        print("Train labels", train_labels)

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        shrinking = [True, False]
        c = [0.5, 1, 1.5, 2, 3]
        gamma = [0.00001, 0.0001, 0.2, 0.3, 'scale', 'auto']
        probability = [True, False]

        if args.kernels is not None:
            kernels = args.kernels
        if args.shringing is not None:
            shrinking = args.shrinking
        if args.c is not None:
            c = args.c
        if args.gamma is not None:
            gamma = args.gamma
        if args.probability is not None:
            probability = args.probability

        param_grid = dict(C=c, kernel=kernels, shrinking=shrinking, gamma=gamma, probability=probability)
        grid = GridSearchCV(self.clf, param_grid, cv=2, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid.fit(self.mega_histogram, train_labels)

        print(grid.best_params_)
        print(grid.best_score_)

        self.clf = grid.best_estimator_

        counter = len([name for name in os.listdir("./") if os.path.isfile(name)])
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv("SVM_results"+str(counter), columns=["mean_fit_time", "param_kernel", "param_shrinking", "param_C", "param_gamma", "param_probability", "mean_test_neg_mean_squared_error", "rank_test_neg_mean_squared_error", "mean_test_r2", "rank_test_r2"],
                       header=["mean_fit_time", "kernel", "shrinking", "C", "gamma", "probability", "mean_test_nmse", "rank_test_nmse", "mean_test_r2", "rank_test_r2"])

        solver = ['lbfgs', 'sgd', 'adam']
        activation = ['relu']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        batch_size = [1, 2]
        hidden_layer_sizes = [(100,), (100, 80, 50), (150, 100, 70)]

        if args.solver is not None:
            solver = args.solver
        if args.activation is not None:
            activation = args.activation
        if args.learning_rate is not None:
            learning_rate = args.learning_rate
        if args.batch_size is not None:
            batch_size = args.batch_size
        if args.hidden_layer_sizes is not None:
            hidden_layer_sizes = args.hidden_layer_sizes

        param_grid = dict(solver=solver, activation=activation, hidden_layer_sizes=hidden_layer_sizes, learning_rate=learning_rate, batch_size=batch_size)
        grid = GridSearchCV(self.clf2, param_grid, cv=2, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error', n_jobs=3, verbose=1)
        grid.fit(self.mega_histogram, train_labels)

        print(grid.best_params_)
        print(grid.best_score_)

        self.clf2 = grid.best_estimator_

        counter = len([name for name in os.listdir("./") if os.path.isfile(name)])
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv("MLP_results"+str(counter), columns=["mean_fit_time", "param_solver", "param_activation", "param_learning_rate", "param_batch_size", "param_hidden_layer_sizes", "mean_test_neg_mean_squared_error", "rank_test_neg_mean_squared_error", "mean_test_r2", "rank_test_r2"],
                       header=["mean_fit_time", "solver", "activation", "learning_rate", "batch_size", "hidden_layer_sizes", "mean_test_nmse", "rank_test_nmse", "mean_test_r2", "rank_test_r2"])

        print("Training completed")

    def predict(self, iplist):
        predictions1 = self.clf.predict(iplist)
        predictions2 = self.clf2.predict(iplist)
        return predictions1, predictions2

    def score(self, predictions, truth, classes):
        print(predictions)
        print(truth)
        acc = accuracy_score(predictions, truth)
        print(acc)
        cnf_matrix = confusion_matrix(truth, predictions)
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes, title=str('Confusion matrix, acc='+str(acc)))
        plt.show()

    def plotHist(self, vocabulary = None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.tight_layout()