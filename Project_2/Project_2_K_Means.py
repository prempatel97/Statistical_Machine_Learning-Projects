import scipy.io as sio
import scipy.spatial.distance as distance
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_graph(values, strategy):
    """
    Function to Plot Graph for cluster 2 to 10
    """
    plt.plot(values[0],values[1])
    plt.scatter(values[0],values[1],color='red')
    plt.xlabel('Number of Clusters k ')
    plt.ylabel('Objective Function Value')
    plt.title('K-Means')
    plt.show()

def furthest_far_point(data, new_c):
    """
    Function to choose initial centroids from given samples using furthest point approach.
    """
    C = []
    C.append(random.choice(data))

    for i in range(1,new_c):
        temp = []
        for p in data:
            count = 0
            dist = 0.0
            for c in C:
                count+=1
                dist += distance.euclidean(p,c)
            dist = dist / count
            temp.append(dist)
        pos = temp.index(max(temp))
        C.append(data[pos])
    return C

def nearest_cluster(data, new_c):
    """
    Function to choose initial centroids randomly from given samples.
    """
    return np.asarray(random.sample(list(data), new_c))

def objective(clusters, mean):
    """
    Function to compute objective function for given number of cluster
    """
    object_sum = 0
    cluster_sum = 0
    for k in range(len(clusters)):
        cluster_sq = np.square(clusters[k] - mean[k])
        cluster_sum = np.sum(cluster_sq)
        object_sum += cluster_sum
    return object_sum

def new_centroids(clusters, new_c, old_centroids):
    """
    Finding new centroids for modified clusters.
    """
    cen = []
    for i,cluster in enumerate(clusters):
        if cluster:
            cen.append(np.mean(cluster, axis=0))
        else:
            cen.append(old_centroids[i])
    return np.array(cen)

def min_dist(p, centroids, new_c):
    """
    Computing minimum distance from given point to any centroid point.
    """
    target = 0
    min_d = np.linalg.norm(p - centroids[0])
    for i in range(1,new_c):
        dist = np.linalg.norm(p - centroids[i])
        if (dist <= min_d):
            min_d = dist
            target = i
    return target

def kMeans(arr, n_iter,rand_cent,strategy):
    """
    Implement K means algorithm.
    """
    obj_plot = [[],[]]
    clust_count = 0
    print("Strategy %d"%(strategy))
    for k in range(2,11):
        iter = 0
        r = True

        centroids = rand_cent(arr,k)
        while(r):
            clust_points = [[] for i in range(k)]
            centroids_before = np.array(centroids)
            # For each data point, computing distance from all centroids.
            for i in arr:
                label = min_dist(i,centroids,len(centroids))
                clust_points[label].append(i)
            centroids = new_centroids(clust_points,k,centroids_before)
            err = np.sum(np.square(centroids_before - centroids))
            if iter >= n_iter or err == 0:
                r = False
                clust_count+=1
            else:
                iter += 1
        obj = objective(clust_points,centroids)
        print("Object_Value = %f"%(obj))
        if clust_count < k:
            obj_plot[0].append(k)
            obj_plot[1].append(obj)
        else:
            obj_plot = []
            clust_count = 0
    plot_graph(obj_plot,strategy)

def main():
    d = sio.loadmat('AllSamples.mat')
    dat = np.array(d['AllSamples'])

    # Implement K-means with random initial centers.
    kMeans(dat,50,nearest_cluster,1)
    #kMeans(dat,100,nearest_cluster,1)      #Used for Instance 2

    # Implement K-means with furthest initial centers.
    kMeans(dat,50,furthest_far_point,2)
    #kMeans(dat,100,furthest_far_point,2)   #Used for Instance 2

if __name__ == '__main__':
    main()