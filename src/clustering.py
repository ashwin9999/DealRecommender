import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_dataset(name):

    #loads data from csv file
    my_data = genfromtxt(name, delimiter=',')
    return my_data


def euclidian(a, b):

    #returns the euclidean distance between two points
    return np.linalg.norm(a-b)


def plot(dataset, history_centroids, belongs_to):

    #colors for each centroid cluster
    colors = ['r', 'g']

    #split graph by its axis and actual plot
    fig, ax = plt.subplots()

    #for each point in our dataset
    for index in range(dataset.shape[0]):
        #get all the points assigned to a cluster
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        #assign each datapoint in that cluster a color and plot it
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    #logging the history of centroids calculated via training
    history_points = []
    #for each centroid ever calculated
    for index, centroids in enumerate(history_centroids):
        #print them all out
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)


def kmeans(k, epsilon=0, distance='euclidian'):
    #list to store past centroid values
    history_centroids = [] #not needed for algorithm, but useful in visualizing
    #we are using euclidean distance for this case, can use Manhattan distance
    if distance == 'euclidian':
        dist_method = euclidian
    #set the dataset
    dataset = load_dataset('test.csv')
    #number of rows(num_instances), number of columns(num_features)
    num_instances, num_features = dataset.shape
    #set a random number of clusters(between 0 and n0. of rows-1) of size k [k centroids]
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    #set these to list of past centroids (to visualize progress over time)
    history_centroids.append(prototypes)
    #to keep track of centroids at every iteration
    prototypes_old = np.zeros(prototypes.shape)
    #stores clusters
    belongs_to = np.zeros((num_instances, 1))
    #finds euclidean distance
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        #computing distance between our prototypes and the previous prototypes
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        #for each row(instance) in the dataset
        for index_instance, instance in enumerate(dataset):
            #define a distance vector of size k
            dist_vec = np.zeros((k, 1))
            #for each centroid value [prototype contains centroid values]
            for index_prototype, prototype in enumerate(prototypes):
                #compute distance between data point and every other centroid
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            #find the smallest distance, assign that distance to a cluster
            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        #create a temporary prototype list to store to the history
        tmp_prototypes = np.zeros((k, num_features))

        #for each cluster (k of them)
        for index in range(len(prototypes)):
            #get all the points assigned to a cluster
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            #find the mean of those points, this will be the new centroid
            prototype = np.mean(dataset[instances_close], axis=0)
            #add the new centroid to the new temporary list
            tmp_prototypes[index, :] = prototype

        #set the new list to the current list
        prototypes = tmp_prototypes

        #add the calculated centroids to the history list for plotting
        history_centroids.append(tmp_prototypes)

    #plot(dataset, history_centroids, belongs_to)

    #return calculated centroids, history of all of them, and assignments for which cluster does each datapoint belong to
    return prototypes, history_centroids, belongs_to


def execute():
    dataset = load_dataset('test.csv')
    centroids, history_centroids, belongs_to = kmeans(2)
    plot(dataset, history_centroids, belongs_to)


execute()


