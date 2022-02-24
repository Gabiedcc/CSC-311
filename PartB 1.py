import item_response
from utils import *
import matplotlib.pyplot as plt

import numpy as np

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(matrix, theta, beta, discrim, c):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    matrix_copy = matrix.toarray()
    theta = np.reshape(theta, (1, 542))
    theta_matrix = np.transpose(np.tile(theta, (1774, 1)))
    beta = np.reshape(beta, (1, 1774))
    beta_matrix = np.tile(beta, (542, 1))
    discrim = np.reshape(discrim, (1, 1774))
    discrim_matrix = np.tile(discrim, (542, 1))
    e = np.exp(discrim_matrix *(theta_matrix - beta_matrix))
    log1 = np.log(c * np.ones((542, 1774)) + e)
    log2 = np.log((np.ones((542, 1774)) + e))
    vectorized_int = np.vectorize(int)
    nan_position = np.isnan(matrix_copy)

    matrix_copy[nan_position] = 0
    nan_position = np.ones((542, 1774)) - vectorized_int(nan_position)
    result_matrix = matrix_copy *log1 -nan_position * log2 + (nan_position - matrix_copy) * np.log(1 - c)
    result = np.dot(result_matrix, np.ones((1774, 1)))
    result = sum(result)
    return result

def update_theta_beta(matrix, lr, theta, beta, discrim, c):
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
    matrix_copy = matrix.toarray()
    vint = np.vectorize(int)
    nan_entries = np.isnan(matrix_copy)
    not_nan = np.ones((542, 1774)) - vint(nan_entries)
    matrix_copy[nan_entries] = 0
    theta_matrix = np.transpose(np.tile(theta, (1774, 1)))
    beta_matrix = np.tile(beta, (542, 1))
    discrim_matrix = np.tile(discrim, (542, 1))

    e = np.exp(discrim_matrix * (theta_matrix - beta_matrix))
    target = matrix_copy * e * discrim_matrix /(c * np.ones((542, 1774)) + e) - \
            not_nan * e * discrim_matrix / (np.ones((542, 1774)) + e)
    partial_theta = np.sum(target, axis=1)
    theta += lr * partial_theta

    theta_matrix = np.transpose(np.tile(theta, (1774, 1)))
    e = np.exp(discrim_matrix * (theta_matrix - beta_matrix))
    target = matrix_copy * e * discrim_matrix /(c * np.ones((542, 1774)) + e) - \
         not_nan * e * discrim_matrix / (np.ones((542, 1774)) + e)
    partial_beta = np.sum(target, axis=0)
    beta += (-1) * lr * partial_beta

    beta_matrix = np.tile(beta, (542, 1))
    e = np.exp(discrim_matrix * (theta_matrix - beta_matrix))
    target2 = matrix_copy *(theta_matrix - beta_matrix) * e /(c * np.ones((542, 1774)) + e) \
        -not_nan * (theta_matrix - beta_matrix) * e / (np.ones((542, 1774)) + e)
    partial_discrim = np.sum(target2, axis=0)
    discrim += lr * partial_discrim
    return theta, beta, discrim

def irt(data, val_data, lr, iterations):
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
    # TODO: Initialize theta and beta.
    x = np.random.randint(9, size=542)
    theta = np.array([float(i) for i in x])
    x = np.random.randint(9, size=1774)
    beta = np.array([float(i) for i in x])

    discrim = np.random.uniform(low=0.0, high=1.0, size=1774)

    c = 0.03528208
    val_acc_lst = []
    iteration_list = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, discrim=discrim, c=c)
        score = evaluate(data=val_data, theta=theta, beta=beta, discrim=discrim, c=c)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, discrim = update_theta_beta(data, lr, theta, beta, discrim, c)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, discrim, val_acc_lst

def evaluate(data, theta, beta, discrim, c):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = discrim[q] * (theta[u] - beta[q])
        p_a = c + (1 - c) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    test_data = load_public_test_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.05
    iterations = 49
    np.random.seed(1005705621)
    theta, beta, discrim, val_acc_lst = irt(sparse_matrix, val_data, lr, iterations)
    accurracy =  evaluate(data=test_data, theta=theta, beta=beta, discrim=discrim, c=0.03528208)
    print("test accuracy ", accurracy)
    question = np.random.randint(low=0, high=1773, size=1)

    #for q in question:
    #    beta_q = beta[q]
    #    discrim_q = discrim[q]
    #    theta_list = range(-5, 6)
    #    p_x = lambda x: np.exp(x - beta_q) / (1 + np.exp(x - beta_q))
    #    p = [p_x(i) for i in theta_list]
    #    plt.plot(theta_list, p, label="old: questionID: "+ str(q))
    #    p_x2 = lambda x: np.exp(discrim_q * (x - beta_q)) * (1 - 0.03528208)/(1 + np.exp(discrim_q * (x - beta_q))) + 0.03528208
    #    new_p = [p_x2(i) for i in theta_list]
    #    plt.plot(theta_list, new_p, label="new: questionID: "+ str(q))
    #plt.legend()
    #plt.savefig("Probability given question" + ".png")
    theta, beta, val_acc_lst2 = item_response.irt(train_data, val_data, lr, iterations)
    iteration_list = np.arange(0, 49)
    plt.plot(iteration_list,  val_acc_lst, "b-", label="new model accuracy")
    plt.plot(iteration_list, val_acc_lst2, "b--", label="old model accuracy")
    plt.legend()
    plt.savefig("negative loglikelihood" + ".png")





if __name__ == "__main__":

    main()




