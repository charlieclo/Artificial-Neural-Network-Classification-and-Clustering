import tensorflow as tf
import numpy as np
import csv
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Opening File for Classification
def load_data_classification(path):
    dataset = []
    feature = []
    result = []

    with open(path) as file:
        reader = csv.reader(file)
        next(reader)

        #Feature Selection
        for row in reader:
            feature_list = []
            feature_list.append(row[5])
            feature_list.append(row[6])
            feature_list.append(row[7])
            feature_list.append(row[8])
            feature_list.append(row[9])
            feature_list.append(row[10])
            feature_list.append(row[12])
            feature_list.append(row[14])
            feature_list.append(row[15])
            feature_list.append(row[17])
            feature_list.append(row[18])
            feature_list.append(row[19])
            feature_list.append(row[20])
            feature_list.append(row[21])
            feature_list.append(row[23])

            feature.append(feature_list)
            result.append(row[4])

    #Preprocessed Feature with PCA
    pca_feature = feature_preprocessing(feature)
    #Preprocessed Result
    new_result = result_preprocessing(result)

    #Combine Feature with Result
    for i in range(len(new_result)):
        dataset.append((pca_feature[i], new_result[i]))

    return dataset

#Feature Preprocessing
def feature_preprocessing(feature):
    #Making Selected Feature into Numpy Array
    np_feature = np.array(feature)
    print("Numpy Succeed")

    #Create a Scaler Object
    sc = StandardScaler()
    #Fit the Scaler to The Features and Transform
    std_feature = sc.fit_transform(np_feature)
    print("Standard Succeed")

    #Create a PCA Object with 5 Principal Components
    pca = PCA(n_components=5)
    #Fit PCA and Transform Standar Data
    pca_feature = pca.fit_transform(std_feature)
    print("PCA Succeed")

    return pca_feature

#Result List
result_list = ["a", "b", "c", "d", "e"]

#Result Preprocessing
def result_preprocessing(result):
    preprocessed_result = []

    for r in result:
        result_index = result_list.index(r)
        new_result = np.zeros(len(result_list), 'int')
        new_result[result_index] = 1

        preprocessed_result.append(new_result)

    return preprocessed_result

#Load Data
dataset = load_data_classification("O192-COMP7117-AS01-00-classification.csv")

#Shuffle Data
random.shuffle(dataset)

#Split Data
count = int(.7 * len(dataset))
train_data = dataset[:count]
count2 = int(.2 * len(dataset))
validation_data = dataset[count:count + count2]
test_data = dataset[count + count2:]

#Initialization of Input, Output, and Hidden Layer
num_of_input = 5
num_of_output = 5
num_of_hidden = [3, 2]

#Make Input and Target for Training and Validating
feature_input = tf.placeholder(tf.float32, [None, num_of_input])
target = tf.placeholder(tf.float32, [None, num_of_output])

#Initialization of Learning Rate, Number of Epoch
learning_rate = .1
num_of_epoch = 5000

#Save Directory and Filename
save_dir = "./bpnn-model"
filename = "/bpnn.ckpt"

#FullyConnected -> Activcation Function
def fully_connected(input, num_of_input, num_of_output):
    #Make Weight and Bias
    w = tf.Variable(tf.random_normal([num_of_input, num_of_output]))
    b = tf.Variable(tf.random_normal([num_of_output]))
    Wx_b = tf.matmul(input, w) + b
    act = tf.nn.sigmoid(Wx_b)

    return act

def build_model(input):
    layer1 = fully_connected(input, num_of_input, num_of_hidden[0])
    layer2 = fully_connected(layer1, num_of_hidden[0], num_of_hidden[1])
    layer3 = fully_connected(layer2, num_of_hidden[1], num_of_output)
    
    return layer3

#Training
def optimize(model, train_data, validation_data):
    validating = 0
    error = tf.reduce_mean(.5 * (target - model) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(error)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
    #Change Data Type
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        #Running Session
        sess.run(tf.global_variables_initializer())
        
        #Saver to Save Model
        saver = tf.train.Saver(tf.global_variables())
        
        for epoch in range(1, num_of_epoch + 1):
            #Inputting Training Feature, Result, and Feed
            train_feature = [data[0] for data in train_data]
            train_result = [data[1] for data in train_data]
            train_feed = { feature_input: train_feature, target: train_result }

            #Inputting Validation Feature, Result, and Feed
            validation_feature = [data[0] for data in validation_data]
            validation_result = [data[1] for data in validation_data]
            validation_feed = { feature_input: validation_feature, target: validation_result }

            #Training
            _, train_loss, train_accuracy = sess.run([optimizer, error, accuracy], train_feed)

            #Printing Training Result every 100 Epoch
            if epoch % 100 == 0:
                print("Training Epoch   : {:5} with Error: {:1.8f}".format(epoch, train_loss))

            #Validating Data and Save its Model when Lower than Previous Error
            if epoch % 500 == 0:
                #Validating
                validation_loss, validation_accuracy = sess.run([error, accuracy], validation_feed)

                if validating == 0:
                    min_validation_loss = validation_loss
                    validating = 1
                    print("Validation Epoch : {:5} with Error: {:1.8f} and Accuracy {:1.8f}, saving the model".format(epoch, validation_loss, validation_accuracy))
                    saver.save(sess, save_dir+filename, epoch)

                else:
                    if min_validation_loss > validation_loss:
                        print("Validation Epoch : {:5} with Error: {:1.8f} and Accuracy {:1.8f}, saving the model".format(epoch, validation_loss, validation_accuracy))
                        saver.save(sess, save_dir+filename, epoch)

                    else:
                        print("Validation Epoch : {:5} with Error: {:1.8f} and Accuracy {:1.8f}, not saving the model".format(epoch, validation_loss, validation_accuracy))

def testing(model, test_data):
    error = tf.reduce_mean(.5 * (target - model) ** 2)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
    #Change Data Type
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        #Running Session
        sess.run(tf.global_variables_initializer())

        #Inputting Testing Feature, Result and Feed
        test_feature = [data[0] for data in test_data]
        test_result = [data[1] for data in test_data]
        test_feed = { feature_input: test_feature, target: test_result }
        #Testing
        test_loss, test_accuracy = sess.run([error, accuracy], test_feed)

        #Printing Testing Result
        print("\nTesting Success with Error: {:1.8f} and Accuracy: {:1.8f}".format(test_loss, test_accuracy))

model = build_model(feature_input)
optimize(model, train_data, validation_data)
testing(model, test_data)