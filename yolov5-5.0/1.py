import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np
# import pytorch

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
plt.plot(X[:, 0], X[:, 1], 'b.')
plt.show()

dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)
print(dbscan.labels_[:10])
print(dbscan.core_sample_indices_[:10])
print(np.unique(dbscan.labels_))
dbscan2 = DBSCAN(eps=0.05, min_samples=5)
dbscan2.fit(X)



def plot_dbscan(dbscan, X, size, show_xlables=True, show_ylables=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    # print(core_mask[:100]) 核心点
    core_mask[dbscan.core_sample_indices_] = True
    # print(core_mask) 离群点
    anomalies_mask = dbscan.labels_ == -1
    # print(anomalies_mask)
    non_core_mask = ~(core_mask | anomalies_mask)
    # print(non_core_mask)

    cores=dbscan.components_
    anomalies=X[anomalies_mask]
    non_cores=X[non_core_mask]

    # plt.scatter(cores[:,0],cores[:,1],c=dbscan.labels_[core_mask],
    #             marker='o',s=size,cmap="Paired")
    plt.scatter(cores[:,0],cores[:,1],marker='*',
                s=20,c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:,0],anomalies[:,1],marker='x',s=100,c='r')
    plt.scatter(non_cores[:, 0], non_cores[:, 1], marker='.',c=dbscan.labels_[non_core_mask])
    if show_xlables:
        plt.xlabel("$x_1$",fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylables:
        plt.ylabel("$x_2$",fontsize=14,rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f},min_sample={}".format(dbscan.eps,dbscan.min_samples),fontsize=14)

plt.figure(figsize=(9,4))

plt.subplot(121)
plot_dbscan(dbscan,X,size=600)

plt.subplot(122)
plot_dbscan(dbscan2,X,size=100)

plt.show()




