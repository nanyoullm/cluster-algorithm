# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 簇对象
class Cluster(object):
    def __init__(self, name):
        self.name = name
        self.points = []

    def has_point(self, point):
        return point in self.points

    def add_point(self, point):
        self.points.append(point)

    def get_point_x(self):
        return list(map(lambda x: x[0], self.points))

    def get_point_y(self):
        return list(map(lambda x: x[1], self.points))

    def point_count(self):
        return len(self.points)


# DBScan对象
class DBScan(object):
    def __init__(self, min_points, eps):
        self.min_points = min_points
        self.eps = eps
        self.visited = []
        self.cluster = []
        self.num_cluster = 0
        self.data = []
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # 计算两点之间的欧氏距离
    @staticmethod
    def cal_distance(p1, p2):
        return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

    # 寻找邻居点
    def find_neighbours(self, point):
        neighbours = []
        for p in self.data:
            if p != point and self.cal_distance(p, point) <= self.eps:
                neighbours.append(p)
        return neighbours

    # 获取指定名称的簇对象
    def get_cluster(self, name):
        for cluster in self.cluster:
            if cluster.name == name:
                return cluster
        raise ValueError('no such cluster named {}'.format(name))

    # 发展簇成员
    def expand_cluster(self, cluster, neighbours):
        for point in neighbours:
            if point not in self.visited:
                self.visited.append(point)
                # 寻找邻居的邻居
                new_neighbours = self.find_neighbours(point)
                if len(new_neighbours) >= self.min_points:
                    for nnp in new_neighbours:
                        if nnp not in neighbours:
                            neighbours.append(nnp)

                # 如果该点没有被其他簇据为己有，自己之前也没发展过，那么就收了它
                for other in self.cluster:
                    if not other.has_point(point):
                        if not cluster.has_point(point):
                            cluster.add_point(point)

        # 发展结束
        self.cluster.append(cluster)

    # 运行dbscan算法
    def run_dbscan(self, data):
        self.data = data
        fig = plt.figure()
        ax = fig.add_subplot(111)

        noise = Cluster('Noise')
        self.cluster.append(noise)

        for point in self.data:
            if point not in self.visited:
                self.visited.append(point)
                neighbours = self.find_neighbours(point)
                if len(neighbours) < self.min_points:
                    noise.add_point(point)
                else:
                    # 到了这里说明发现了一个新的簇
                    cluster_name = 'Cluster_{}'.format(self.num_cluster)
                    new_cluster = Cluster(cluster_name)
                    self.num_cluster += 1

                    new_cluster.add_point(point)
                    # 发展这个簇成员
                    self.expand_cluster(new_cluster, neighbours)
                    # 画出散点图
                    ax.scatter(new_cluster.get_point_x(), new_cluster.get_point_y(),
                               c=self.color[self.num_cluster], marker='o', label=cluster_name)
        if noise.point_count() > 0:
            ax.scatter(noise.get_point_x(), noise.get_point_y(),
                       c=self.color[-1], marker='x', label='Noise')

        ax.legend(loc='best')
        plt.title('DBScan')
        plt.show()


if __name__ == '__main__':
    dbscan = DBScan(min_points=2, eps=2.5)
    # 读取数据并随机打乱
    data = pd.read_csv('data/abc_shuffle.csv', names=['x', 'y', 'label']).sample(frac=1).values
    data = list(map(lambda x: x.tolist(), data))
    dbscan.run_dbscan(data)
    print('end')
