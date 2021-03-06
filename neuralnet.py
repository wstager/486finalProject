import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import sys

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

POL_DICT = {"CNN": 0, "NYT": 0, "Politico": 0, "OccupyDemocrats": 0, "Slate": 0,
            "ABC": 1, "PBS": 1, "USAToday": 1, "NBCNews": 1, "TheHill": 2, "FoxNews": 2, "Breitbart": 2, "Reason": 2, "WashingtonExaminer": 2}

def init_weights(shape):
    """ 
    Weight initialization
    INPUT: (row size, column size) tuple
    OUTPUT: a TensorFlow variable that maintains the state of the neural net weight values
    """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    INPUT: X, an input data matrix
           w_1, weights for connections between input layer and hidden layer
           w_2, weights for connections between hiden layer and output layer
    OUTPUT: an array of prediction values for each political category
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))
    yhat = tf.nn.sigmoid(tf.matmul(h, w_2))
    return yhat

def get_top_adjectives(num_adj):
    """
    Derives lists of the most meaningful ajectives towards identifying the gender of a sentence
    INPUT: num_adj, number of adjectives to record for both the female and male lists
    OUTPUT: female_list, a list of the adjectives that were most informative of the female gender
            male_list, a list of the adjectives that wer most informative of the male gender
    """
    female_list = []
    male_list = []
    with open("adj_files/female_adj.txt") as female_file:
        adj_list = [line.rstrip('\n') for line in female_file]
        adj_list = adj_list[1:num_adj + 1]
        female_list = [x.split('\t')[0] for x in adj_list]
    with open("adj_files/male_adj.txt") as male_file:
        adj_list = [line.rstrip('\n') for line in male_file]
        adj_list = adj_list[1:num_adj + 1]
        male_list = [x.split('\t')[0] for x in adj_list]
    return female_list, male_list

def get_site_adjectives(sitename):
    """
    Computes counts of the informative adjectives seen in each article in the given news site.
    INPUT: sitename, the name of the site to compute the adjective feature values for
    OUTPUT: male_adj_dict, a dictionary mapping from url name to dictionary with informative adjectives as keys and counts as values
            female_adj_dict, a dictionary mapping from url name to dictionary with informative adjectives as keys and counts as values
    """
    male_adj_dict = {}
    female_adj_dict = {}
    current_url = ""
    current_gender = ""
    with open("adj_files/ADJ_{}.txt".format(sitename)) as adj_file:
        adj_list = [line.rstrip('\n') for line in adj_file]
        for adj_item in adj_list:
            first_item, second_item = adj_item.strip().split('\t')
            if first_item == "male":
                male_adj_dict[second_item] = {}
                current_url = second_item
                current_gender = "male"
                continue
            if first_item == "female":
                female_adj_dict[second_item] = {}
                current_url = second_item
                current_gender = "female"
                continue
            if current_gender == "female":
                female_adj_dict[current_url][first_item] = second_item
            elif current_gender == "male":
                male_adj_dict[current_url][first_item] = second_item
    return male_adj_dict, female_adj_dict


def get_data():
    """
    Creates feature and label matrices for each url in our data set
    OUTPUT: all_X, matrix that has a row of feature values for each article
            all_Y, matrix that has a row of values indicating the label of the article (e.g. [1, 0, 0] for liberal)
            num_columns, the number of features used for each article
    """
    all_X = ""
    all_Y = ""
    first = True
    row = 0
    num_top_adjectives = 40
    ftop_adjs, mtop_adjs = get_top_adjectives(num_top_adjectives)
    for filename in os.listdir("feat_files"):
        sitename = filename.split('_')[1][:-4]
        male_adj_dict, female_adj_dict = get_site_adjectives(sitename)
        with open("feat_files/{}".format(filename)) as inputfile:
            feature_lists = [line.rstrip('\n') for line in inputfile]
            feature_lists = feature_lists[1:]
            num_rows = len(feature_lists)
            num_columns = len(feature_lists[0].strip().split('\t')) - 3
            num_columns += num_top_adjectives * 2 #change if we
            if first:
                all_X = np.ones((num_rows, num_columns + 1))
                all_Y = np.zeros((num_rows, 3)) # liberal, neutral, conservative
                first = False
            else:
                all_X = np.concatenate((all_X, np.ones((num_rows, num_columns + 1))), axis=0)
                all_Y = np.concatenate((all_Y, np.zeros((num_rows, 3))), axis=0)
            for feature_list in feature_lists:
                feature_list = feature_list.strip().split('\t')
                f_num = 0
                url = feature_list[0].strip()
                for feature in feature_list[3:]:
                    all_X[row, f_num] = float(feature)
                    f_num += 1
                for adj in ftop_adjs:
                    if adj in female_adj_dict[url]:
                        all_X[row, f_num] = float(female_adj_dict[url][adj])
                    else:
                        all_X[row, f_num] = 0
                    f_num += 1
                for adj in mtop_adjs:
                    if adj in male_adj_dict[url]:
                        all_X[row, f_num] = float(male_adj_dict[url][adj])
                    else:
                        all_X[row, f_num] = 0
                    f_num += 1
                all_Y[row, POL_DICT[sitename]] = 1
                row += 1
    return all_X, all_Y, num_columns

def test_NN(h_size):
    """
    Trains and tests the neural network on 4 different folds of the dataset.
    Prints accuracy, precision, and recall for each test of a different split of the data.
    Also prints average values for each of the above metrics across the four tests.
    INPUT: h_size, the number of nodes to include in the hidden layer of the neural net
    """
    all_X, all_Y, num_columns = get_data()
    # generate vector of y lables
    y = np.zeros(len(all_Y))
    for y_row in range(0, len(all_Y)):
        if all_Y[y_row][0]:
            y[y_row] = 0
        elif all_Y[y_row][1]:
            y[y_row] = 1
        elif all_Y[y_row][2]:
            y[y_row] = 2

    skf = StratifiedKFold(n_splits=4)
    train_X = ""
    train_Y = ""
    test_X = ""
    test_Y = ""
    average_accuracy = 0
    average_lib_prec = 0
    average_lib_rec = 0
    average_con_prec = 0
    average_con_rec = 0
    average_n_prec = 0
    average_n_rec = 0
    for train_indices, test_indices in skf.split(all_X, y):
        num_train_rows = len(train_indices)
        num_test_rows = len(test_indices)
        train_X = np.zeros((num_train_rows, num_columns + 1))
        train_Y = np.zeros((num_train_rows, 3))
        test_X = np.zeros((num_test_rows, num_columns + 1))
        test_Y = np.zeros((num_test_rows, 3))
        row = 0
        for tr_index in train_indices:
            train_X[row] = all_X[tr_index]
            train_Y[row] = all_Y[tr_index]
            row += 1
        row = 0
        for t_index in test_indices:
            test_X[row] = all_X[t_index]
            test_Y[row] = all_Y[t_index]
            row += 1
        accuracy, prec_lib, prec_n, prec_con, recall_lib, recall_n, recall_con = train_NN(h_size, train_X, train_Y, test_X, test_Y)
        print("accuracy {} \n liberal precision {} \n liberal recall {} \n conservative precision {} \n conservative recall {} \n neutral precision {} \n neutral recall {}".format(accuracy, prec_lib, recall_lib, prec_con, recall_con, prec_n, recall_n))
        average_accuracy+=accuracy
        average_lib_prec+=prec_lib
        average_lib_rec+=recall_lib
        average_con_prec+=prec_con
        average_con_rec+=recall_con
        average_n_prec+=prec_n
        average_n_rec+=recall_n

    average_accuracy = float(average_accuracy) / float(4)

    average_lib_prec = float(average_lib_prec) / float(4)
    average_con_prec = float(average_con_prec) / float(4)
    average_n_prec = float(average_n_prec) / float(4)

    average_lib_rec = float(average_lib_rec) / float(4)
    average_con_rec = float(average_con_rec) / float(4)
    average_n_rec = float(average_n_rec) / float(4)

    print("average accuracy {} \n average liberal precision {} \n average liberal recall {} \n average conservative precision {} \n average conservative recall {} \n average neutral precision {} \n average neutral recall {}".format(average_accuracy, average_lib_prec, average_lib_rec, average_con_prec, average_con_rec, average_n_prec, average_n_rec))


def train_NN(h_size, train_X, train_y, test_X, test_y):
    """
    Trains the neural network on the training data set and tests on the input test set
    INPUT: h_size, the number of nodes to include in the hidden layer of the neural net
           train_X, a matrix with the feature values for each article in the training set
           train_y, a matrix with the label values for each article in the training set
           test_X, a matrix with the feature values for each article in the test set
           test_y, a matrix with the label values for each article in the test set
    """
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 262 features and 1 bias
    y_size = train_y.shape[1]   # Number of outcomes: 3 (liberal, neutral, conservative)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    best_test_accuracy = 0
    best_prec_lib = 0
    best_prec_n = 0
    best_prec_con = 0
    best_recall_lib = 0
    best_recall_n = 0
    best_recall_con = 0

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_run = sess.run(predict, feed_dict={X: train_X, y: train_y})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 train_run)
        test_run = sess.run(predict, feed_dict={X: test_X, y: test_y})
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 test_run)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_prec_lib, best_prec_n, best_prec_con, best_recall_lib, best_recall_n, best_recall_con = calc_pr(test_y, test_run)

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()
    return best_test_accuracy, best_prec_lib, best_prec_n, best_prec_con, best_recall_lib, best_recall_n, best_recall_con

def calc_pr(test_y, test_run):
    """
    Calculates precision and recall values for each political category
    INPUT: test_y, a matrix with the ground truth label values for each article in the test set
           test_run, an array of the predicted categories (0: liberal, 1: neutral, 2: conservative) for each test article
    OUTPUT: prec_lib, precision value for liberal category
            prec_n, precision value for neutral category
            prec_con, precision value for conservative category
            recall_lib, recall value for liberal category
            recall_n, recall value for neutral category
            recall_con, recall value for conservative category
    """
    tp_lib, tp_con, tp_n = 0, 0, 0
    fp_lib, fp_con, fp_n = 0, 0, 0
    fn_lib, fn_con, fn_n = 0, 0, 0

    golden_y = np.zeros(len(test_y))

    for y_row in range(0, len(test_y)):
        if test_y[y_row][0]:
            golden_y[y_row] = 0
        elif test_y[y_row][1]:
            golden_y[y_row] = 1
        elif test_y[y_row][2]:
            golden_y[y_row] = 2

    count = 0
    for val in golden_y:
        #liberal
        if int(val) == 0 and test_run[count] == 0:
            tp_lib += 1
        elif int(val) == 0:
            fn_lib += 1
        elif test_run[count] == 0:
            fp_lib += 1
        #neutral
        if int(val) == 1 and test_run[count] == 1:
            tp_n += 1
        elif int(val) == 0:
            fn_n += 1
        elif test_run[count] == 0:
            fp_n += 1
        #conservative
        if int(val) == 2 and test_run[count] == 2:
            tp_con += 1
        elif int(val) == 2:
            fn_con += 1
        elif test_run[count] == 2:
            fp_con += 1
        count+=1

    if fp_lib == 0 and tp_lib == 0:
        prec_lib = 1
    else:
        prec_lib = float(tp_lib)/float(tp_lib+fp_lib)
    if tp_n == 0 and fp_n == 0:
        prec_n = 1
    else:
        prec_n = float(tp_n)/float(tp_n+fp_n)

    if tp_con == 0 and fp_con == 0:
        prec_con = 1
    else:
        prec_con = float(tp_con)/float(tp_con+fp_con)

    if tp_lib == 0 and fn_lib == 0:
        recall_lib = 1
    else:
        recall_lib = float(tp_lib)/float(tp_lib+fn_lib)

    if fn_n == 0 and tp_n == 0:
        recall_n = 1
    else:
        recall_n = float(tp_n)/float(tp_n+fn_n)

    if fn_con == 0 and tp_con == 0:
        recall_con = 1
    else:
        recall_con = float(tp_con)/float(tp_con+fn_con)

    return prec_lib, prec_n, prec_con, recall_lib, recall_n, recall_con

def main():
    h_size = int(float(sys.argv[1])) # number of hidden nodes
    test_NN(h_size)


if __name__ == '__main__':
    main()
