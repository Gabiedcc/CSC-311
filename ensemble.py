# TODO: complete this file.
import numpy as np
import random
from item_response import *
from knn import *
#from neural_network import *
from sklearn.impute import KNNImputer
from utils import *
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.utils.data
from part_a.item_response import *


def bootstrap():
    train_dic = load_train_csv("../data")
    l1 = []
    l2 = []
    for i in range(3):
        size = len(train_dic["user_id"])
        user = []
        question = []
        correct = []
        m = np.zeros([542, 1774]) * np.nan
        for i in range(56688):
            idx = random.randint(0,size-1)
            u = train_dic["user_id"][idx]
            user.append(u)
            q = train_dic["question_id"][idx]
            question.append(q)
            c = train_dic["is_correct"][idx]
            correct.append(c)
            if np.isnan(m[u][q]):
                m[u][q]=c
            else:
                m[u][q]+=c
        train_d = {"user_id":user,"question_id":question,"is_correct":correct}
        l1.append(train_d)
        l2.append(m)
    return l1,l2

def irt_prediction(data, theta, beta):
    prob_list = []
    for t in range(len(data["is_correct"])):
        theta_i = theta[data["user_id"][t]]
        beta_j = beta[data["question_id"][t]]
        prob = sigmoid(theta_i - beta_j)
        prob_list.append(prob)
    return np.array(prob_list)

def prediction(prob_list):
    a = []
    for i in prob_list:
        if i >= 0.5:
            a.append(1)
        else:
            a.append(0)
    return np.array(a)

def bias(prediction, true_labels):
    squared_diff = (prediction - true_labels) * (prediction - true_labels)
    bias = sum(squared_diff) / len(squared_diff)
    return bias

def variance(model, bagged_prediction):
    squared_diff = np.zeros((len(model), ))
    squared_diff += (model - bagged_prediction) * (model - bagged_prediction)
    squared_diff = squared_diff / len(model)
    variance = sum(squared_diff) / len(squared_diff)
    return variance

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    true_label = np.array(val_data["is_correct"])
    np.random.seed(1005705621)
    lr = 0.025
    iterations = 20
    theta, beta, val_acc = irt(train_data, val_data, lr, iterations)
    original = irt_prediction(val_data, theta, beta)
    resampled, matrix = bootstrap()
    theta, beta, val_acc = irt(resampled[0], val_data, lr, iterations)
    print("original accuracy: ", val_acc[-1])
    ensemble1 = irt_prediction(val_data, theta, beta)
    theta, beta, val_acc = irt(resampled[1], val_data, lr, iterations)
    ensemble2 = irt_prediction(val_data, theta, beta)
    theta, beta, val_acc = irt(resampled[2], val_data, lr, iterations)
    ensemble3 = irt_prediction(val_data, theta, beta)
    print("ensemble accuracy: ", val_acc[-1])
    ensemble = (ensemble1 + ensemble3 + ensemble2) / 3
    y0 = prediction(original)
    y1 = prediction(ensemble1)
    y2 = prediction(ensemble2)
    y3 = prediction(ensemble3)
    y4 = prediction(ensemble)
    y_bar = (y0 + y1 + y2 + y3 + y4) / 5
    bias_original = bias(y0, true_label)
    bias_ensemble = bias(y4, true_label)
    print("original bias: ", bias_original)
    print("ensemble bias: ", bias_ensemble)

    variance_original = variance(y0, y_bar)
    variance_ensemble = variance(y4, y_bar)
    print("original variance: ", variance_original)
    print("ensemble variance: ", variance_ensemble)


    #Test set:
    print("test set: ")
    true_label = np.array(test_data["is_correct"])
    original = irt_prediction(test_data, theta, beta)
    ensemble1 = irt_prediction(test_data, theta, beta)
    ensemble2 = irt_prediction(test_data, theta, beta)
    ensemble3 = irt_prediction(test_data, theta, beta)
    ensemble = (ensemble1 + ensemble3 + ensemble2) / 3
    y0 = prediction(original)
    y1 = prediction(ensemble1)
    y2 = prediction(ensemble2)
    y3 = prediction(ensemble3)
    y4 = prediction(ensemble)
    y_bar = (y0 + y1 + y2 + y3 + y4) / 5
    bias_original = bias(y0, true_label)
    bias_ensemble = bias(y4, true_label)
    acc = 0
    for i in range(len(y0)):
        if y0[i] == true_label[i]:
            acc += 1
    test_acc = acc / len(true_label)
    print("original model accuracy on test set: ", test_acc)
    acc = 0
    for i in range(len(y0)):
        if y4[i] == true_label[i]:
            acc += 1
    test_acc = acc / len(true_label)
    print("ensemble model accuracy on test set: ", test_acc)
    print("original bias: ", bias_original)
    print("ensemble bias: ", bias_ensemble)

    variance_original = variance(y0, y_bar)
    variance_ensemble = variance(y4, y_bar)
    print("original variance: ", variance_original)
    print("ensemble variance: ", variance_ensemble)
    #np.savetxt("original predict in test.txt", y0, fmt="%s")
    #np.savetxt("ensemble predict on test.txt ", y4, fmt="%s")
    #np.savetxt("avg prediction.txt", y_bar, fmt="%s")



if __name__ == "__main__":
    main()
