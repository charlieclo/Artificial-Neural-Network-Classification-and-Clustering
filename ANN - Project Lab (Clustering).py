import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Opening File for Clustering
def load_data_clustering(path):
    dataset = []

    with open(path) as file:
        reader = csv.reader(file)
        next(reader)

        #Feature Selection
        for row in reader:
            calorie_density = (float(row[3]) + float(row[4])) / 100
            dividend = float(row[8]) +  float(row[9])
            divisor = float(row[7]) + float(row[11])
            if divisor == 0:
                divisor = 1
            fat_ratio = dividend / divisor
            sugar = float(row[14])
            protein = float(row[16])
            salt = float(row[17])

            dataset.append([calorie_density, fat_ratio, sugar, protein, salt])

    return dataset

class Clustering:
    def __init__(self, height, width, input_dimension):
        #Initialization
        self.height = height
        self.width = width
        self.input_dimension = input_dimension

        #Node Initialization
        self.node = [tf.to_float([i, j]) for i in range(height) for j in range(width)]
        self.input = tf.placeholder(tf.float32, [input_dimension])
        self.weight = tf.Variable(tf.random_normal([height * width, input_dimension]))

        #Function to Find Nearest Node and Weight Update
        self.best_matching_unit = self.get_bmu(self.input)
        self.updated_weight = self.get_update_weight(self.best_matching_unit, self.input)

    def get_bmu(self, input):
        #Calculate Shortest Distance using Euclidean
        expand_input = tf.expand_dims(input, 0)
        euclidean = tf.square(tf.subtract(expand_input, self.weight))
        distances = tf.reduce_sum(euclidean, 1) #Putting 0 will be Adding to Beside

        #Search Index of Shortest Distance
        minimum_index = tf.argmin(distances, 0) #Putting 0 will be Checking to Down | Putting 1 will be Checking to Beside
        bmu_location = tf.stack([tf.mod(minimum_index, self.width), tf.div(minimum_index, self.width)])
        return tf.to_float(bmu_location)

    def get_update_weight(self, bmu, input):
        #Learning Rate and Sigma Initialization
        learning_rate = 0.5
        sigma = tf.to_float(tf.maximum(self.height, self.width) / 2)

        #Calculate Distance between BMU to Other Node using Euclidean
        expand_bmu = tf.expand_dims(bmu, 0)
        euclidean = tf.square(tf.subtract(expand_bmu, self.node))
        distances = tf.reduce_sum(euclidean, 1)
        ns = tf.exp(tf.negative(tf.div(distances**2, 2 * sigma**2)))

        #Calculate Rate
        rate = tf.multiply(ns, learning_rate)
        num_of_node = self.height * self.width
        rate_value = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.input_dimension]) for i in range(num_of_node)])

        #Calculate Update Weight
        weight_difference = tf.multiply(rate_value, tf.subtract(tf.stack([input for i in range (num_of_node)]), self.weight))

        updated_weight = tf.add(self.weight, weight_difference)
        return tf.assign(self.weight, updated_weight)

    def train(self, dataset, num_of_epoch):
        with tf.Session() as sess:
            #Running Session
            sess.run(tf.global_variables_initializer())
            
            #Clustering
            for i in range (num_of_epoch):
                print(i)
                for data in dataset:
                    feed = { self.input : data}
                    sess.run([self.updated_weight], feed)
            self.weight_value  = sess.run(self.weight)
            self.node_value = sess.run(self.node)
            cluster = [ [] for i in range(self.width)]
            for i, location in enumerate(self.node_value): #Return Index
                cluster[int(location[0])].append(self.weight_value[i])
            self.cluster = cluster

#Load Data
dataset = load_data_clustering("O192-COMP7117-AS01-00-clustering.csv")

#Making Selected Feature into Numpy Array
np_dataset = np.array(dataset)

#Create a Scaler Object
sc = StandardScaler()
#Fit the Scaler to The Features and Transform
std_dataset = sc.fit_transform(np_dataset)

#Create a PCA Object with 3 Principal Components
pca = PCA(n_components=3)
#Fit PCA and Transform Standar Data
pca_dataset = pca.fit_transform(std_dataset)

#SOM
som = Clustering(10, 10, 3)
som.train(pca_dataset, 5000)
plt.imshow(som.cluster)
plt.show()
