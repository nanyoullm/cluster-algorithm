# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 造数据
x = [1.0, 1.3, 1.9, 1.5, 2.4, 2.0, 2.3, 2.9, 2.5, 1.4,
     4.6, 3.9, 5.1, 4.7, 4.2, 5.6, 5.9, 5.3, 5.7, 5.2,
     7.1, 7.5, 8.9, 8.0, 7.7, 8.1, 8.5, 7.9, 8.5, 8.7]

y = [1.1, 1.4, 2.0, 1.8, 1.5, 1.4, 1.6, 2.1, 1.6, 1.8,
     2.6, 2.2, 2.7, 2.6, 2.3, 2.6, 2.2, 2.7, 2.9, 2.3,
     1.4, 1.1, 2.4, 1.7, 2.0, 1.7, 1.5, 2.1, 2.2, 2.0]
color = ['r', 'b', 'y']

cluster_x = np.random.randint(0, 8, 3).tolist()
cluster_y = np.random.randint(1, 3, 3).tolist()

# plt.figure(1, figsize=(7, 7))
# plt.plot(x, y, 'ko', markerfacecolor='none')
# plt.plot(cluster_x, cluster_y, 'o')
# plt.title('data')

fig = plt.figure(1, figsize=(7, 7))
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.scatter(cluster_x, cluster_y)
plt.title('data')


def cal_distance(x1, x2, y1, y2):
    return np.sqrt(pow(x1-x2, 2)+pow(y1-y2, 2))


def kmeans(x, y, cluster_x, cluster_y):
    new_belong_x = []
    new_belong_y = []
    new_center_x = []
    new_center_y = []

    for i in range(len(cluster_x)):
        new_belong_x.append([])
        new_belong_y.append([])

    for i in range(len(x)):
        distance = np.array([])
        for j in range(len(cluster_x)):
            distance = np.append(distance, cal_distance(x[i], cluster_x[j], y[i], cluster_y[j]))
        belong = np.argmin(distance)
        new_belong_x[belong].append(x[i])
        new_belong_y[belong].append(y[i])

    for i in range(len(cluster_x)):
        if len(new_belong_x[i]) != 0:
            new_center_x.append(np.array(new_belong_x[i]).mean())
            new_center_y.append(np.array(new_belong_y[i]).mean())
        else:
            new_center_x.append(cluster_x[i] + 1)
            new_center_y.append(cluster_y[i] + 1)

    return new_belong_x, new_belong_y, new_center_x, new_center_y


plt.ion()
plt.draw()
plt.pause(1)
for i in range(5):
    plt.clf()
    print('epoch: {}'.format(i))
    new_belong_x, new_belong_y, center_x, center_y = kmeans(x, y, cluster_x, cluster_y)
    for j in range(len(new_belong_x)):
        plt.plot(center_x[j], center_y[j], '{}x'.format(color[j]), markersize=12)
        plt.plot(new_belong_x[j], new_belong_y[j], '{}o'.format(color[j]))
    plt.draw()
    plt.savefig('{}.jpg'.format(i))
    plt.pause(0.5)
    cluster_x = center_x
    cluster_y = center_y

plt.ioff()
plt.show()
