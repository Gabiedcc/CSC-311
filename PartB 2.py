from utils import *
import matplotlib.pyplot as plt

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return 1.2**(x) / (1 + 1.2**x)


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for t in range(len(data['is_correct'])):
        i = data['user_id'][t]
        j = data['question_id'][t]
        c_ij = data['is_correct'][t]
        log_lklihood += c_ij * (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} sparse matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    t = 12
    user_copy = np.array(data['user_id'])
    question_copy = np.array(data['question_id'])
    partial_theta = []
    for i in range(542):
        j_list = np.where(user_copy == i)
        j_list = [int(j) for j in j_list[0]]
        partial_theta_i = 0
        for j in j_list:
            question = data['question_id'][j]
            partial_theta_i +=  np.exp(t/10)*((data['is_correct'][j]-1)*10.**beta[question]*t**theta[i]+data['is_correct'][j]*10.**theta[i]*t**beta[question])/(10.**theta[i]*t**beta[question]+10.**beta[question]*t**theta[i])
        partial_theta.append(partial_theta_i)
    theta += lr * np.array(partial_theta)
    partial_beta = []
    for j in range(1774):
        i_list = np.where(question_copy == j)
        i_list = [int(i) for i in i_list[0]]
        partial_beta_j = 0
        for i in i_list:
            student = data['user_id'][i]
            partial_beta_j += -np.exp(t/10) * ((data['is_correct'][i] - 1) * 10. ** beta[j]* t ** theta[student]+ data['is_correct'][
                i] * 10. ** theta[student]* t ** beta[j]) / (
                        10. ** theta[student] * t ** beta[j] + 10. ** beta[j] * t ** theta[student])

        partial_beta.append(partial_beta_j)
    beta += lr * np.array(partial_beta)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data,lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    x = np.random.randint(9, size=542)
    theta = np.array([float(i) for i in x])
    x = np.random.randint(9, size=1774)
    beta = np.array([float(i) for i in x])
    question_meta = load_question_meta()
    user_category = np.zeros([542,1774])


    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)
    betamean = beta.mean()
    user_copy = np.array(data['user_id'])
    for i in range(542):
        j_list = np.where(user_copy == i)
        j_list = [int(j) for j in j_list[0]]
        total = 0.
        for j in j_list:
            question = data['question_id'][j]
            total+=beta[question]
            for k in question_meta[question]:
                tmp = int(k.strip())
                user_category[i][tmp] +=1
        theta[i] += (total/len(j_list)-betamean)*2


    return theta, beta,user_category


def evaluate(data, theta, beta,user_category):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    question_category = load_question_meta()
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        regularization = 0.
        for k in question_category[q]:
            tmp = int(k.strip())
            regularization+=user_category[u][tmp]
        x+=regularization*0.0004
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_copy = sparse_matrix.toarray()



    #Bootstrap:
    #training1, training2, training3 = bootstrap()
    #theta, beta, val_acc_lst


    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.025
    iterations = 20
    np.random.seed(1005705621)
    neg_lld_list = []
    neg_lld_val_list = []
    iteration_list = []
    x = np.random.randint(9, size=542)
    theta = np.array([float(i) for i in x])
    x = np.random.randint(9, size=1774)
    beta = np.array([float(i) for i in x])
    t,b,u = irt(train_data,  lr, iterations)
    print(evaluate(val_data,t,b,u))
    # for i in range(iterations + 1):
    #     neg_lld = -neg_log_likelihood(train_data, theta=theta, beta=beta)
    #     neg_lld_val = - neg_log_likelihood(val_data, theta=theta, beta=beta)
    #     neg_lld_list.append(neg_lld)
    #     neg_lld_val_list.append(neg_lld_val)
    #     theta, beta = update_theta_beta(train_data, lr, theta=theta, beta=beta)
    #     iteration_list.append(iterations)
    #     score = evaluate(data=val_data, theta=theta, beta=beta)
    #     print("NLLK: {} \t Score: {}".format(neg_lld, score))

    #plt.plot(iteration_list,  neg_lld_list, "b-", "training loglikelihood")
    #plt.show()
    #plt.plot(iteration_list, neg_lld_val_list, "b--", "validation loglikelihood")
    #plt.show()
    to = 0.
    # a = beta.max()
    # b = beta.min()
    # print(a)
    # print(b)
    # print(beta.mean())
    # print(theta.min())
    # print(theta.max())
    # print(theta.mean())
    score = evaluate(val_data, t, b,u)
    print("Validation accuracies is ", score)
    score = evaluate(test_data, t, b,u)
    print("Test Accuracies is ", score)

    question = np.random.randint(low=0, high=1773, size=5)
    for q in question:
        beta_q = beta[q]
        theta_list = range(-5, 6)
        p_x = lambda x: np.exp(x - beta_q) / (1 + np.exp(x - beta_q))
        p = [p_x(i) for i in theta_list]
        plt.plot(theta_list, p, label="questionID: "+ str(q))
    plt.legend()
    plt.savefig("Probability given 5 questions" + ".png")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################




if __name__ == "__main__":
    main()
