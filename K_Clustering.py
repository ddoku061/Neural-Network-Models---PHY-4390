#Code by Doga Dokuz, 2025
import numpy as np
import matplotlib.pyplot as plt


with open('q3_training_data.npy', 'rb') as f:
    data = np.load(f)
k = 5  #number of clusters

def centroids_generator(k, data):
    #as asked in the question, we want to initialize k number of centroids randomly
    np.random.seed(42)  # or 50 found this online, reference to machine learning random number generator
    centroids = np.random.uniform(-5, 5, size=(k, data.shape[1])) #generating centroids [-5,5]
    return centroids

def find_clusters(data, centroids):
    #we want to assign each data point to the nearest centroid, by calculating their distance to the centroid
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # data[] is our training data, we are calculating the distances between the data points and centroids with matrix norm function
    cluster_index = np.argmin(distances, axis=1)  #finding the index of the clusters with minimal distance
    return cluster_index

def update_centroids(data, cluster_indices, k):
    #each centroid will be updated to the mean of all training examples
    new_centroids = np.zeros((k, data.shape[1])) #initiating a matrix the same shape as the data
    
    for i in range(k): #we have k=5 so 5 cluster points
        cluster_points = data[cluster_indices == i] 
        #each centroid is updated to the mean of all the training examples assinged to the cluster i
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0) #taking the mean along each x and y dimensions so assigning the mean as the new centroid for that specific cluster
    return new_centroids

def k_means_clustering(data, k):
    centroids = centroids_generator(k, data) #takes the data and places random centroids in the graph
    # prev_centroids = np.zeros_like(centroids)
    centroid_changes = []

    for iteration in range(100): #limiting the maximum iteration count with 100
        cluster_indices = find_clusters(data, centroids) #finding clusters depending on the closest/smallest distance to the dataset
        new_centroids = update_centroids(data, cluster_indices, k)
        
        #calculating the change in the distance between new and old centroids to check the convergence later
        change = np.linalg.norm(new_centroids - centroids)
        centroid_changes.append(change) #storing the changes to plot them
        
        plot(data, cluster_indices, centroids, iteration)

        #if the change is small enough, we can stop the algorithm
        if change < 0.0001: #we can't make it 0, since we want to avoid division by numerical 0
            print(f"Converged after {iteration + 1} iterations.")
            break
        #we update our centroids with the new centroids
        centroids = new_centroids

    return cluster_indices, centroids, centroid_changes

def plot(data, cluster_indices, centroids, iteration):
    #plotting clusters and corresponding centroids
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'magenta', 'purple', 'yellow', 'cyan']
    
    for i in range(len(centroids)): #going through all the k points
        cluster_points = data[cluster_indices == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f"Cluster {i+1}")
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
    
    plt.title(f"Unsupervised Learning: K-Means Clustering \n Iteration {iteration + 1}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    
    plt.legend()
    plt.grid()
    plt.show()

cluster_indices, final_centroids, changes = k_means_clustering(data, k)

#plotting the change in distance of centroids, theoretically needs to get smaller at each iteration because of the convergence
plt.figure(figsize=(8, 6))
plt.plot(changes, marker='o')
plt.title("Change in Centroid Positions")
plt.xlabel("Iteration")
plt.ylabel("|L^2| of Centroid Changes")
plt.grid()
plt.show()
