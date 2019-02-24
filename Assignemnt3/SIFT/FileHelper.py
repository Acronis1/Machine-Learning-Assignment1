import glob
import cv2
from numpy import loadtxt
import os


class FileHelper:

    def __init__(self):
        self.trainImglist = {}
        self.testImglist = {}
        self.trainImgCount = 0
        self.testImgCount = 0

    def getFruitFiles(self, split):
        """
        - returns  a dictionary of all files
        having key => value as  objectname => image path
        - returns total number of files.
        """
        fruit_names = loadtxt(os.path.join("fruitData", "fruits.txt"), delimiter=";", unpack=False, dtype='str')
        count = 0
        for fruit in fruit_names:
            print(" #### Reading image category ", fruit, " ##### ")
            path = fruit
            files = glob.glob(os.path.join("fruitData", path, '*.jpg'))
            self.trainImglist[fruit] = []
            self.testImglist[fruit] = []
            splitpoint = len(files)*split
            for i, imagefile in enumerate(files):
                print("Reading file ", imagefile)
                im = cv2.imread(imagefile, 0)
                if i <= splitpoint:
                    self.trainImglist[fruit].append(im)
                    self.trainImgCount += 1
                else:
                    self.testImglist[fruit].append(im)
                    self.testImgCount += 1

    def getCarFiles(self):
        """
        - returns  a dictionary of all files
        having key => value as  objectname => image path
        - returns total number of files.
        """
        path = "carData"
        paths = ["TrainImages", "TestImages"]
        self.trainImglist["neg"] = []
        self.testImglist["pos"] = []
        self.trainImglist["pos"] = []
        self.testImglist["neg"] = []
        count = 0
        for path in paths:
            print(" #### Reading train Images from: ", path, " ##### ")
            files = glob.glob(os.path.join("carData", path, '*.pgm'))
            for i, imagefile in enumerate(files):
                print("Reading file ", imagefile)
                im = cv2.imread(imagefile, 0)
                if im is None:
                    continue
                if "neg" in imagefile:
                    cls = "neg"
                else:
                    cls = "pos"
                if path == "TrainImages":
                    self.trainImglist[cls].append(im)
                    self.trainImgCount += 1
                else:
                    self.testImglist[cls].append(im)
                    self.testImgCount += 1

    def getTrainData(self):
        return [self.trainImglist, self.trainImgCount]

    def getTestData(self):
        return [self.testImglist, self.testImgCount]
