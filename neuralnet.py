import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

POL_DICT = {"CNN": 0, "NYT": 0, "Politico": 0, "OccupyDemocrats": 0, "Slate": 0,
            "ABC": 1, "PBS": 1, "USAToday": 1, "NBCNews": 1, "TheHill": 2, "FoxNews": 2, "Breitbart": 2, "Reason": 2, "WashingtonExaminer": 2}
def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    #print("target: ", target)

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    
    #print("all X: ", all_X)

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    # print("all Y: ", all_Y)
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def test_get_data():
    with open("feat_NYT.txt", 'r') as inputfile:
        feature_lists = [line.rstrip('\n') for line in inputfile]
        feature_lists = feature_lists[1:]
        num_rows = len(feature_lists)
        num_columns = 6
        all_X = np.ones((num_rows, num_columns + 1))
        all_Y = np.zeros((num_rows, 3)) # liberal, neutral, conservative
        row = 0
        for feature_list in feature_lists:
            feature_list = feature_list.split('\t')
            all_X[row, 1] = float(feature_list[3]) #female sent
            all_X[row, 2] = float(feature_list[4]) #male sent
            all_X[row, 3] = float(feature_list[8]) #affect female
            all_X[row, 4] = float(feature_list[81]) #affect male
            all_X[row, 5] = float(feature_list[35]) #sexual female
            all_X[row, 6] = float(feature_list[108]) #sexula male
            all_Y[row, 0] = 1
            row += 1
        with open("feat_FoxNews.txt", 'r') as inputfile:
            feature_lists = [line.rstrip('\n') for line in inputfile]
            feature_lists = feature_lists[1:]
            num_rows = len(feature_lists)
            all_X = np.concatenate((all_X, np.ones((num_rows, num_columns + 1))), axis=0)
            all_Y = np.concatenate((all_Y, np.zeros((num_rows, 3))), axis=0)
            for feature_list in feature_lists:
                feature_list = feature_list.split('\t')
                all_X[row, 1] = float(feature_list[3]) #female sent
                all_X[row, 2] = float(feature_list[4]) #male sent
                all_X[row, 3] = float(feature_list[8]) #affect female
                all_X[row, 4] = float(feature_list[81]) #affect male
                all_X[row, 5] = float(feature_list[35]) #sexual female
                all_X[row, 6] = float(feature_list[108]) #sexula male
                all_Y[row, 2] = 1
                row += 1
        return train_test_split(all_X, all_Y, test_size=0.3, random_state=RANDOM_SEED)

def get_top_adjectives(num_adj):
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
    all_X = ""
    all_Y = ""
    first = True
    row = 0
    num_top_adjectives = 20
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
                url = feature_list[0]
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
    return train_test_split(all_X, all_Y, test_size=0.3, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes: 1 (the score)

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

    for epoch in range(200):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

    
if __name__ == '__main__':
    main()



