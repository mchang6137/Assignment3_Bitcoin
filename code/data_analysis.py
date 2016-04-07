import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import *
from scipy.sparse import csr_matrix

DATA_PATH = "../data/%s"


def get_train_data(fname):
    return pd.read_csv(DATA_PATH % fname,
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])


def get_aggr(X, key, val, frac):
    grouped = X[[key, val]].groupby(key).sum()
    grouped = grouped.sample(frac = frac)
    # plt.hist(grouped[val].values, 100)
    # plt.show()
    return grouped

def send_vs_receive_stat(X):
    senders =  X["sender"].unique()
    receivers = X["receiver"].unique()
    print "# senders :", senders.shape[0]
    print "# receivers :", receivers.shape[0]
    intersect = np.intersect1d(senders, receivers)
    print "# senders and receivers", intersect.shape[0]

    key = "receiver"
    grouped_recv = X[[key, "transaction"]].groupby(key, as_index = False).sum()
    grouped_recv = grouped_recv.sort_values(by = "transaction", ascending = False)
    grouped_recv.columns = ["id", "receives"]
    
    key = "sender"
    grouped_send = X[[key, "transaction"]].groupby(key, as_index = False).sum()
    grouped_send = grouped_send.sort_values(by = "transaction", ascending = False)
    grouped_send.columns = ["id", "sends"]

    merged = pd.merge(grouped_send, grouped_recv, left_on = "id", right_on = "id")
    # merged = merged[(merged.sends > 100) &  (merged.receives > 100)]
    print merged.shape
    sends = merged['sends'].as_matrix()
    recvs = merged['receives'].as_matrix()
    print "send/receive correlation :", np.corrcoef(sends, recvs)[0, 1]

    # for K in range(50, 2000, 100):
    #     topk_recv = grouped_recv.head(K)
    #     topk_send = grouped_send.head(K)    
            
    #     intersect = np.intersect1d(topk_send["id"].as_matrix(), topk_recv["id"].as_matrix())
    #     print K, ":", ("%0.2f" % (len(intersect) / float(K)))

    # K = 100
    # topk_recv = grouped_recv.head(K)    
    # joined = pd.merge(topk_recv, grouped_send, left_on = "id", right_on = "id")
    
    # plt.plot(joined["receives"].values, "r-")
    # plt.plot(joined["sends"].values, "b-")

    # topk_send = grouped_send.head(K)
    # joined = pd.merge(topk_send, grouped_recv, left_on = "id", right_on = "id")
    
    # plt.plot(joined["receives"].values, "g-")
    # plt.plot(joined["sends"].values, "y-")

    # plt.show()

def PCA_analysis(train):
    shape = (train['sender'].max() + 1, train['receiver'].max() + 1)

    train_csr = csr_matrix((train['transaction'], (train['sender'], train['receiver'])),
                           shape = shape)

    svd = TruncatedSVD(n_components=2)
    X = svd.fit_transform(train_csr)
    # print X
    # print(svd.explained_variance_ratio_) 

    shape = (train['receiver'].max() + 1, train['sender'].max() + 1)

    train_csr = csr_matrix((train['transaction'], (train['receiver'], train['sender'])),
                           shape = shape)

    svd = TruncatedSVD(n_components=2)
    X = svd.fit_transform(train_csr)
    # print X
    # print(svd.explained_variance_ratio_) 

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    
if __name__ == "__main__":
    train = get_train_data("txTripletsCounts.txt")
    PCA_analysis(train)