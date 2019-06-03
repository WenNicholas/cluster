# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip() # feature_type为tfidf

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix

#第一步：读取数据
book_data = pd.read_csv('data/data.csv') #读取文件
#,sep=",",error_bad_lines=False,engine='python',encoding='utf-8'
print(book_data.head())  #2822 rows x 5 columns
# print("csv数据",os.linesep,book_data)

book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()

print('书名:', book_titles[0])
print('内容:', book_content[0][:10]) #内容前10个字

#第二步：数据载入、分词
from normalization import normalize_corpus

# normalize corpus
norm_book_content = normalize_corpus(book_content)  #返回的是分词后的集合['现代人 内心 流失 的 东西……','在 第一次世界大战 的……'，……]

# 第三步：提取 tf-idf 特征
vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                  feature_type='tfidf',
                                                  min_df=0.2, max_df=0.90,
                                                  ngram_range=(1, 2))
# 查看特征数量
print(feature_matrix.shape)  #得到tf-idf矩阵，稀疏矩阵表示法  (2822, 16281) 2822行，16281个词汇（根据16281个词汇建立词的索引）
"""
  (0, 11185)	0.17921956529547814   表示为：第0个列表元素，**词典中索引为11185的元素**， 权值0.17921956529547814
  (0, 3199)	0.1644425715576606
  (0, 10416)	0.21232583039538178
  (0, 1451)	0.15573088636535332
"""
# 获取特征名字
feature_names = vectorizer.get_feature_names() #显示所有文本的词汇，列表类型
# print(vectorizer.vocabulary_)    #词汇表，字典类型
# print(feature_matrix.toarray())   #.toarray() 是将结果转化为稀疏矩阵
# 打印某些特征
print(feature_names[:10]) #显示前10个文本的词汇，列表类型


#第四步：提取完特征后，进行聚类
from sklearn.cluster import KMeans

#KMeans++
def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)  #km打印结果是KMeans的参数
    km.fit(feature_matrix)
    clusters = km.labels_  #编号 [4 6 6 ... 2 2 2]
    return km, clusters

#设置k=10,聚出10个类别
num_clusters = 10
km_obj, clusters = k_means(feature_matrix=feature_matrix,
                           num_clusters=num_clusters)
book_data['Cluster'] = clusters  #在原先的csv文本中加入一列Cluster后的数字
# print(book_data)


#第五步：查看每个cluster的数量
from collections import Counter

# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())
#dict_items([(2, 415), (3, 177), (5, 1084), (9, 221), (4, 285), (0, 47), (7, 168), (8, 66), (6, 269), (1, 90)])


def get_cluster_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books

    return cluster_details

def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties


def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, book_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'title': book_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()



#打印结果
cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

#画出簇类（没跑通）
# plot_clusters(num_clusters=num_clusters,
#               feature_matrix=feature_matrix,
#               cluster_data=cluster_data,
#               book_data=book_data,
#               plot_size=(16, 8))





