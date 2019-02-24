from ImageHelper import *
from FileHelper import *
from BOVHelper import *
import argparse
import time


class BOV:
    def __init__(self, no_clusters, args):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelper()
        self.bov_helper = BOVHelper(no_clusters)
        self.file_helper = FileHelper()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.labels = {}
        self.dataSet = None
        self.split = 0
        self.args = args


    def loadData(self,category="fruit", split=0.8):
        # read file. prepare file lists.
        if category == "car":
            self.file_helper.getCarFiles()
        else:
            self.file_helper.getFruitFiles(split)


    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """
        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getTrainData()

        start_time = time.time()

        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("Computing Features for ", word+": "+str(len(imlist))+" images, with label: "+str(label_count))
            self.labels[word] = label_count
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                kp, des = self.im_helper.features(im)
                if len(kp) == 0:
                    #imlist.remove(im)
                    self.trainImageCount -= 1
                    continue
                self.train_labels = np.append(self.train_labels, label_count)
                self.descriptor_list.append(des)

            label_count += 1

        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        # self.bov_helper.plotHist()

        self.bov_helper.standardize()


        print("--- %s seconds for feature extraction---" % (time.time() - start_time))
        input("Press enter to continue")

        self.bov_helper.train(self.train_labels, self.args)


    def recognize(self, test_img, test_image_path=None):

        """
        This method recognizes a single image
        It can be utilized individually as well.
        """

        kp, des = self.im_helper.features(test_img)
        # print kp

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        #print(vocab)
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        lb1, lb2 = self.bov_helper.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb1, lb2

    def testModel(self):
        """
        This method is to test the trained classifier
        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image
        """

        self.testImages, self.testImageCount = self.file_helper.getTestData()

        predictions1 = []
        predictions2 = []
        truth = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in imlist:
                truth.append(self.labels[word])
                # print imlist[0].shape, imlist[1].shape
                cl1, cl2 = self.recognize(im)
                predictions1.append(cl1)
                predictions2.append(cl2)

        self.bov_helper.score(predictions1, truth, self.labels.keys())
        self.bov_helper.score(predictions2, truth, self.labels.keys())
        # for each in predictions:
        #     # cv2.imshow(each['object_name'], each['image'])
        #     # cv2.waitKey()
        #     # cv2.destroyWindow(each['object_name'])
        #     #
        #     plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
        #     plt.title(each['object_name'])
        #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SVM and MLP classifying with SIFT features")

    # general args:
    parser.add_argument('--split', action="store", dest="split", required=False, help="Train/Test split")
    parser.add_argument('-d', action="store", dest="data", required=False, help="data set to train on")

    # SVM args
    parser.add_argument('-k', action="store", dest="kernels", required=False, choices=['linear', 'poly', 'rbf', 'sigmoid'], help="SVM kernel")
    parser.add_argument('-s', action="store_true", dest="shrinking", required=False, help="SVM Shrinking heuristic true/false")
    parser.add_argument('-c', action="store", dest="c", type=float, required=False, help="SVM Penalty parameter C of the error term")
    parser.add_argument('-g', action="store", dest="gamma", type=float, required=False, help="SVM Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’")
    parser.add_argument('-p', action="store_true", dest="probability", required=False, help="SVM Whether to enable probability estimates")

    # MLP args:
    parser.add_argument('-slv', action="store", dest="solver", required=False, choices=['lbfgs', 'sgd', 'adam'], help="MLP The solver for weight optimization")
    parser.add_argument('-a', action="store", dest="activation", required=False, choices=['relu', 'logistic', 'tanh', 'identity'], help="MLP activation function for the hidden layer")
    parser.add_argument('-l', action="store", dest="learning_rate", required=False, choices=['constant', 'invscaling', 'adaptive'], help="MLP learning_rate")
    parser.add_argument('-b', action="store", dest="batch_size", type=int, required=False, help="MLP Size of minibatches for stochastic optimizers")
    parser.add_argument('-hl', action="store", dest="hidden_layer_sizes", type=int, required=False, help="MLP number of neurons for hidden layer")

    args = parser.parse_args()
    print(args)

    bov = BOV(100, args)

    # # SVM args
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # shrinking = [True, False]
    # c = [0.5, 1, 1.5, 2, 3]
    # gamma = [0.00001, 0.0001, 0.2, 0.3, 'scale', 'auto']
    # probability = [True, False]
    #
    # # MLP args:
    # solver = ['lbfgs', 'sgd', 'adam']
    # activation = ['relu', 'logistic']
    # learning_rate = ['constant', 'invscaling', 'adaptive']
    # batch_size = [1, 2, 3, 5, 10]
    # hidden_layer_sizes = [50, 100, 150]
    data = "fruit"
    split = 0.8
    if args.data is not None:
        data = args.data
    if args.split is not None:
        split = args.split

    bov.loadData(data, split)
    bov.trainModel()
    bov.testModel()
