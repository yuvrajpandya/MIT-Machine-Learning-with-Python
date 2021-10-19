import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        ...where k is number of classes

        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE

    H = np.zeros((theta.shape[0], X.shape[0]))
    intermediate_result = np.zeros((theta.shape[0], X.shape[0]))

    # constant c to prevent numeric underflow/overflow
    c = np.max(np.dot(theta, X.T)/temp_parameter, axis=0)

    # compute the total probability i.e. the sum of all classes j probabilities
    for index_jth_label in range(0, theta.shape[0]):
        intermediate_result[index_jth_label] = np.exp((np.dot(theta[index_jth_label], X.T)/temp_parameter)-c)

    # compute probability of each X[i] datapoint across k classes
    for ith_datapoint in range(0, intermediate_result.shape[1]):
        total_probability = 0
        for jth_label in range(0, intermediate_result.shape[0]):
            total_probability = np.sum(intermediate_result[:, ith_datapoint])
            H[jth_label][ith_datapoint] = intermediate_result[jth_label][ith_datapoint]/total_probability

    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    cost = 0
    epsilon = 1e-6  # to avoid log(0) which is undefined
    n_samples = X.shape[0]
    d_features = X.shape[1]
    k_classes = theta.shape[0]
    intermediate_result = np.zeros((k_classes, n_samples))  # intermediate_result of size (k,n) k classes, n samples

    # compute the total probability i.e. the sum of all classes j probabilities
    for index_jth_label in range(0, k_classes):
        intermediate_result[index_jth_label] = np.exp((np.dot(theta[index_jth_label], X.T) / temp_parameter))

    # compute the error of mis-classifying a data point - for each datapoint.
    error = 0
    for ith_datapoint in range(0, n_samples):  # for each data point i in n samples
        for jth_label in range(0, k_classes):  # for each j class in k classes

            if Y[ith_datapoint] == jth_label:
                total_probability = np.sum(intermediate_result[:, ith_datapoint])
                #  check 0 is not passed in log below
                if intermediate_result[jth_label][ith_datapoint] / total_probability > epsilon:
                    error += np.log(intermediate_result[jth_label][ith_datapoint] / total_probability)
                break

    # compute the regularization penalty
    reg_error = 0
    for jth_label in range(0, k_classes):  # for each k class
        for dth_feature in range(0, d_features):  # for each theta associated with feature
            reg_error += np.square(theta[jth_label][dth_feature])

    cost += (-1/n_samples) * error + (lambda_factor / 2) * reg_error

    return cost

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE

    itemp = 1. / temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1] * num_examples, (Y, range(num_examples))), shape=(num_labels, num_examples)).toarray()
    non_regularized_gradient = np.dot(M - probabilities, X)
    non_regularized_gradient *= -itemp / num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
