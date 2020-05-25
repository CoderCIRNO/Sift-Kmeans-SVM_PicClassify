import cv2
import os
import numpy as np
from tqdm import tqdm
#提取的sift特征数目200
sift_num=100
#k 50
wordCnt = 50
#kmeans终止迭代精度要求0.1
eps=0.1
#kmeans最大迭代次数20
max_iter=20
#kmeans重复次数3
re_kmeans=3

def trainSet2featureSet():
        num_classes = 15
        central_points = []
        #根据文件夹名确定类别名
        classes = os.listdir('./data/')
        class_data_paths = ['./data/'+ i+'/' for i in classes]
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        featureSet = np.float32([]).reshape(0,128)
        for i,path in enumerate(class_data_paths):
                img_names = os.listdir(path)
                #不知道为什么每次都有一个Thumbs.db……
                img_names.remove('Thumbs.db')
                img_names.sort()
                img_names = img_names[:150]
                print(path)
                for img_name in tqdm(img_names):
                        img = cv2.imread(path+img_name,cv2.IMREAD_GRAYSCALE)
                        _, des = SIFT.detectAndCompute(img, None)
                        featureSet = np.append(featureSet, des, axis=0)
                np.save('./features/'+classes[i]+'_feature', featureSet)

def feature2vector(features, centers):
        featVec = np.zeros((1, wordCnt))
        for i in range(0, features.shape[0]):
                fi = features[i]
                diffMat = np.tile(fi, (wordCnt, 1)) - centers
                sqSum = (diffMat**2).sum(axis=1)
                dist = sqSum**0.5
                sortedIndices = dist.argsort()
                idx = sortedIndices[0]
                featVec[0][idx] += 1
        return featVec

def learnVocabulary():
        for name in tqdm(os.listdir('./data')):
                features = np.load("./features/" + name + "_feature.npy")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
                _, labels, centers = cv2.kmeans(features, wordCnt,None, criteria, re_kmeans, cv2.KMEANS_RANDOM_CENTERS)
                filename = "./vocabulary/" + name + ".npy"
                np.save(filename, (labels, centers))

def trainSVM():
        print("生成特征向量")
        trainData = np.float32([]).reshape(0, wordCnt)
        response = np.int32([])
        dictIdx = 0
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        for name in tqdm(os.listdir('./data')):
                class_img_path = "./data/" + name + "/"
                labels, centers = np.load("./vocabulary/" + name + ".npy",allow_pickle=True)
                img_names = os.listdir(class_img_path)
                img_names.remove('Thumbs.db')
                img_names.sort()
                #取前150张做训练集
                img_names = img_names[:150]
                for img_name in img_names:
                        img = cv2.imread(class_img_path+img_name,cv2.IMREAD_GRAYSCALE)
                        _,des = SIFT.detectAndCompute(img, None)
                        featVec = feature2vector(des, centers)
                        trainData = np.append(trainData, featVec, axis=0)
                res = np.repeat(dictIdx, 150)
                response = np.append(response, res)
                dictIdx += 1
        print("训练线性SVM")
        trainData = np.float32(trainData)
        response = response.reshape(-1, 1)
        svm = cv2.ml.SVM_create()
        #线性核SVM
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.train(trainData,cv2.ml.ROW_SAMPLE,response)
        svm.save("svm.clf")

def train():
        trainSet2featureSet()
        learnVocabulary()
        trainSVM()

def test():
        print("测试……")
        svm = cv2.ml.SVM_load("svm.clf")
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        confusion_matrix = []
        for name in tqdm(os.listdir('./data')):
                class_img_path = "./data/" + name + "/"
                labels, centers = np.load("./vocabulary/" + name + ".npy",allow_pickle=True)
                img_names = os.listdir(class_img_path)
                img_names.remove('Thumbs.db')
                img_names.sort()
                img_names = img_names[150:]
                res = [0 for _ in range(15)]
                for img_name in img_names:
                        img = cv2.imread(class_img_path+img_name,cv2.IMREAD_GRAYSCALE)
                        _,des = SIFT.detectAndCompute(img, None)
                        featVec = feature2vector(des, centers)
                        case = np.float32(featVec)
                        dict_svm = svm.predict(case)
                        pred = int(dict_svm[1])
                        res[pred] += 1
                confusion_matrix.append(res)
        for res in confusion_matrix:
                print(res)
        np.save('confusion_matrix', confusion_matrix)

def check_result():
        confusion_matrix = np.load('confusion_matrix.npy',allow_pickle=True)
        for res in confusion_matrix:
            print(res)

if __name__ == "__main__":
        train()
        test()
        #check_result()