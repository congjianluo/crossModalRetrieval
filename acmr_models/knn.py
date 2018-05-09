# -*- coding: utf-8 -*-
import numpy as np

from db_model.db_wikipedia import get_all_wikipedia_dataset

is_init = 0


def init_knn_dataset():
    wikipedia_data = get_all_wikipedia_dataset()
    trains_vecs = []
    trains_label = []
    train_feats = []
    for item in wikipedia_data:
        train_feats.append(np.fromstring(item["feats"], dtype=np.float32))
        trains_vecs.append(np.fromstring(item["vecs"], dtype=np.float32))
        trains_label.append(item["label"])
    return train_feats, trains_vecs, trains_label


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = np.argsort(distance)

    # LimitValue = 10
    # sortedOrdered = [0 for i in range(LimitValue)]
    # sortedOrderedValue = [max(sortedDistIndices) for i in range(LimitValue)]
    #
    # index = 0
    # for item in sortedDistIndices:
    #     for i in range(LimitValue):
    #         if item < sortedOrderedValue[i]:
    #             sortedOrderedValue[i] = item
    #             sortedOrdered[i] = index
    #             break
    #     index += 1
    #
    # print(sortedOrdered)
    # print(sortedOrderedValue)
    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    print(sortedDistIndices[:10])
    return maxIndex, sortedDistIndices[:10]


def kNNClassifyDis():
    return 0


def get_vecs_knn_ret(testX):
    train_feats, trains_vecs, trains_label = init_knn_dataset()
    vecs_groups = trains_vecs
    vecs_groups = np.array([_ for _ in vecs_groups])
    return kNNClassify(testX, vecs_groups, trains_label, 2)


def get_feats_knn_ret(testX):
    train_feats, trains_vecs, trains_label = init_knn_dataset()
    feats_groups = train_feats
    feats_groups = np.array([_ for _ in feats_groups])
    return kNNClassify(testX, feats_groups, trains_label, 2)


if __name__ == "__main__":
    train_feats, trains_vecs, trains_label = init_knn_dataset()
    print(trains_label[2865]["label"])
    tesX = np.fromstring(train_feats[2865]["feats"], dtype=np.float32)
    print(get_feats_knn_ret(tesX))
    tesX = np.fromstring(trains_vecs[2865]["vecs"], dtype=np.float32)
    print(get_vecs_knn_ret(tesX))
