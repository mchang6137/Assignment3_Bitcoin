import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nimfa

import random
from subprocess import call
from sklearn.decomposition import *
from sklearn.metrics import *
from scipy.sparse import csr_matrix

from sklearn.cross_validation import StratifiedKFold, train_test_split
import sklearn.ensemble
import sklearn.discriminant_analysis
import sklearn.tree
import sklearn.linear_model
import sklearn.svm

c_dict = { 'RF' : (sklearn.ensemble, "RandomForestClassifier"),
           'QDA' : (sklearn.discriminant_analysis, "QuadraticDiscriminantAnalysis"),
           'DT' : (sklearn.tree, "DecisionTreeClassifier"),
           'LREG' : (sklearn.linear_model, "LogisticRegression"),
           'LSVM' : (sklearn.svm, "LinearSVC"),
         }

DATA_PATH = "../data/%s"

###################################
#######         data        #######
###################################

def get_data(fname):
    return pd.read_csv(DATA_PATH % fname,
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])

def get_block_data(fname='blockstate.txt'):
    line_count = 0
    address_block_map = {}
    current_address = 0

    with open(fname) as f:
        content = f.readlines()
        for line in content:
            if line_count % 2 == 0:
                current_address = int(line.strip('\n'))
            else:
                address_block_map[current_address] = int(line.strip('\n'))
            line_count += 1

    return address_block_map


def get_zero_sample(train):

    send_max = train[:, 0].max()
    recv_max = train[:, 1].max() 

    train_size = train.shape[0]
    
    zero_samples = np.ndarray(shape = train.shape)
    ind = 0
    while ind < 2:
        s = random.randint(0, send_max)
        r = random.randint(0, recv_max)
        if ((not np.logical_and.reduce([zero_samples[:, 0] == s, zero_samples[:, 1] == r]).any())
            and (not np.logical_and.reduce([train[:, 0] == s, train[:, 1] == r]).any() )):
            zero_samples[ind] = (s, r, 0)
            ind += 1
            print "%d %d 0" % (s, r)


def sample(X, frac):
    length = X.shape[0]
    cnt = int(frac * length)
    inds = np.array(random.sample(range(length), cnt))
    return X[inds]

def sample_cnt(X, cnt):
    length = X.shape[0]
    inds = np.array(random.sample(range(length), cnt))
    return X[inds]

###################################
#######         MF          #######
###################################

def MF_predict(P, Q, test, threshold = None):
    Yres = np.array([np.dot( P[row['sender'],:], Q[:,row['receiver']])  for index,row in test.iterrows()])

    OrgYres = Yres
    if not threshold is None:
        Yres[Yres > threshold] = 1
        Yres[Yres < 1] = 0

    return (Yres, OrgYres)

# 0.000001
def TSVD(train, K, shape = None):
    if shape is None:
        shape = (train['sender'].max() + 1, train['receiver'].max() + 1)

    train_csr = csr_matrix((train['transaction'], (train['sender'], train['receiver'])),
                           shape = shape)

    svd = TruncatedSVD(n_components = K)
    P = svd.fit_transform(train_csr)
    Q = svd.components_
    return (P, Q)

# 0.00001
def NNegMF(train, K, shape = None):
    if shape is None:
        shape = (train['sender'].max() + 1, train['receiver'].max() + 1)

    train_csr = csr_matrix((train['transaction'], (train['sender'], train['receiver'])),
                           shape = shape)

    mf = NMF(n_components = K)
    P = mf.fit_transform(train_csr)
    Q = mf.components_

    return (P, Q)

def ProbMF(train, K, shape = None, frac = 0.5):
    if shape is None:
        shape = (train['sender'].max() + 1, train['receiver'].max() + 1)

    train = sample(train.as_matrix(), frac)
    
    train_csr = csr_matrix((train[:, 2], (train[:, 0], train[:, 1])),
                           shape = shape)

    mf = nimfa.Pmf(train_csr)
    P = mf.factorize()
    print P
    # Q = mf.components_

    # Yres = np.array([np.dot( P[row['sender'],:], Q[:,row['receiver']])  for index,row in test.iterrows()])
    # YTest = test.as_matrix()[:, 2]

    # evaluate(YTest, Yres)

def PF(train, K, shape = None):
    if shape is None:
        n = train['sender'].max() + 1
        m = train['receiver'].max() + 1
    else:
        n = shape[0]
        m = shape[1]

    # call('hgaprec -n %d -m %d -k %d -dir %s/Pdata' % (n, m, K, DATA_PATH), shell = True)

    # hgaprec -n 444075 -m 444065 -k 2 -dir ../data/Pdata

    res_temp = "./n%d-m%d-k%d-batch-vb/%s"

    P = np.ndarray(shape = (n + 1, K))
    P_PF = np.loadtxt(res_temp % (n, m, K, "byusers.tsv"))[:, 1:]
    P[P_PF[:, 0].astype(int)] = P_PF[:, 1:]
    P = P[1:]
    

    Q = np.ndarray(shape = (m + 1, K))
    Q_PF = np.loadtxt(res_temp % (n, m, K, "byitems.tsv"))[:, 1:]
    Q[Q_PF[:, 0].astype(int)] = Q_PF[:, 1:]
    Q = Q[1:]

    return (P, Q.T)

def HPF(train, K, shape = None):
    if shape is None:
        n = train['sender'].max() + 1
        m = train['receiver'].max() + 1
    else:
        n = shape[0]
        m = shape[1]

    # call('hgaprec -hier -m %d -n %d -k %d -dir %s/Pdata' % (m, n, K, DATA_PATH), shell = True)

    # hgaprec -hier -n 444075 -m 444065 -k 2 -dir ../data/Pdata

    res_temp = "./n%d-m%d-k%d-batch-hier-vb/%s" 

    P = np.ndarray(shape = (n + 1, K))
    P_PF = np.loadtxt(res_temp % (n, m, K, "byusers.tsv"))[:, 1:]
    P[P_PF[:, 0].astype(int)] = P_PF[:, 1:]
    P = P[1:]
    

    Q = np.ndarray(shape = (m + 1, K))
    Q_PF = np.loadtxt(res_temp % (n, m, K, "byitems.tsv"))[:, 1:]
    Q[Q_PF[:, 0].astype(int)] = Q_PF[:, 1:]
    Q = Q[1:]

    return (P, Q.T)

###################################
#######      analysis       #######
###################################

def get_aggr(X, key, val, frac):
    grouped = X[[key, val]].groupby(key).sum()
    grouped = grouped.sample(frac = frac)
    plt.hist(grouped[val].values, 100)
    plt.show()

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

    print "intersection of top K senders and receivers"
    for K in range(50, 2000, 100):
        topk_recv = grouped_recv.head(K)
        topk_send = grouped_send.head(K)    
            
        intersect = np.intersect1d(topk_send["id"].as_matrix(), topk_recv["id"].as_matrix())
        print K, ":", ("%0.2f" % (len(intersect) / float(K)))

    # plotting top K senders (receivers) and their corresponding
    # receives (sends).

    K = 100
    topk_recv = grouped_recv.head(K)    
    joined = pd.merge(topk_recv, grouped_send, left_on = "id", right_on = "id")
    
    plt.plot(joined["receives"].values, "r-")
    plt.plot(joined["sends"].values, "b-")

    topk_send = grouped_send.head(K)
    joined = pd.merge(topk_send, grouped_recv, left_on = "id", right_on = "id")
    
    plt.plot(joined["receives"].values, "g-")
    plt.plot(joined["sends"].values, "y-")

    plt.show()

def feature_analysis(train, mf):
    # TODO: create graphs
    (P, Q) = mf(train, 2)
    Q = Q.T

    plt.scatter(P[:, 0], P[:, 1])
    plt.savefig("plots/senders_%s.png" % mf.__name__)

    plt.clf()

    plt.scatter(Q[:, 0], Q[:, 1])
    plt.savefig("plots/receivers_%s.png" % mf.__name__)
    

###################################
#######      classify       #######
###################################

def select(selector_info, **kwargs):
    (module, selector) = selector_info
    sel = getattr(module, selector)
    s = sel(**kwargs)
    return s

def get_classifier_X_Y(train, zeros, test, mf, K1, K2,
                        shortest_path_train = None, shortest_path_test = None, 
                        use_blockstate = False):    
    
    use_shortest_path = not ((shortest_path_train is None) or (shortest_path_test is None))

    (P, Q) = mf(train, max(K1, K2))
    Q = Q.T

    sender = P[:, 0 : (K1 + 1)]
    recv = Q[:, 0 : (K2 + 1)]

    train = train.as_matrix()
    nonzero_trainsize = train.shape[0]
    zeros = zeros.as_matrix()
    zero_trainsize = zeros.shape[0]
    test = test.as_matrix()

    train = np.concatenate((train, zeros), axis = 0)
    sender_ids = sender[train[:, 0]]
    recv_ids = recv[train[:, 1]]
    X = np.append(sender_ids, recv_ids, 1)
    
    if use_shortest_path:
        shortest_path_train = shortest_path_train.as_matrix()
        shortest_path_test = shortest_path_test.as_matrix()
        shortest_path_test_size = shortest_path_test.shape[0]

        train_shortest_path_list = []
        #Iterate through the 1 training set and assign the shortest path to 0
        for index in range(0, nonzero_trainsize):
            train_shortest_path_list.append(1)
        for index in range(0, zero_trainsize):
            if shortest_path_train[index,2] == -1:
                train_shortest_path_list.append(10)
            else:
                train_shortest_path_list.append(int(shortest_path_train[index,2]))
        # print train_shortest_path_list
        
        train_shortest_path_list = np.array(train_shortest_path_list)
        train_shortest_path_list = np.transpose(train_shortest_path_list)
        X= np.c_[X, train_shortest_path_list]
        # print X
    
    if use_blockstate:
        blockstate = get_block_data()

        sender_blockstates = []
        for address in train[:,0]:
            sender_blockstates.append(blockstate[address])
        receiver_blockstates = []
        for address in train[:,1]:
            receiver_blockstates.append(blockstate[address])

        #sender_ids = np.c_[sender_ids, np.array(sender_blockstates)]
        #recv_ids = np.c_[recv_ids, np.array(receiver_blockstates)]
        
    Y = train[:, 2]
    Y[Y > 0] = 1
    
    #Testing Data Manipulation
    sender_ids = sender[test[:, 0]]
    recv_ids = recv[test[:, 1]]

    if use_blockstate:
        sender_blockstates = []
        for address in test[:,0]:
            sender_blockstates.append(blockstate[address])
        receiver_blockstates = []
        for address in test[:,1]:
            receiver_blockstates.append(blockstate[address])
        
        #sender_ids = np.c_[sender_ids, np.array(sender_blockstates)]
        #recv_ids = np.c_[recv_ids, np.array(receiver_blockstates)]
    
    XTest = np.append(sender_ids, recv_ids, 1)

    if use_shortest_path:
        #Need to find the shortest path for the training set too!
        test_shortest_path_list = []
        for index in range(0, shortest_path_test_size):
            if shortest_path_test[index,2] == -1:
                test_shortest_path_list.append(10)
            else:
                test_shortest_path_list.append(int(shortest_path_test[index,2]))
        test_shortest_path_list = np.array(test_shortest_path_list)
        #Change here to include the shortest paths between the sender and receiver
        XTest = np.c_[XTest, test_shortest_path_list]

    return (X, Y, XTest)

def classify(X, Y, XTest, classifier_info, **kwargs):
    # QDA
    predictor = select(classifier_info, **kwargs)
    predictor = predictor.fit(X, Y)

    Yres = predictor.predict(XTest)
    Yprobs = predictor.predict_proba(XTest)[:, 1]
    # Yprobs = Yres

    return (Yres, Yprobs)

###################################
#######      evaluate       #######
###################################

def metrics_str(metrics):
    metrics = ["%.2f" % k for k in metrics]
    return " & ".join(metrics)

def evaluate(YTest, Yres, Yprobs):
    (fpr, tpr, thresholds) = roc_curve(YTest, Yprobs)
    area = auc(fpr, tpr)
    precision_macro = precision_score(YTest, Yres, average="binary")
    recall_macro = recall_score(YTest, Yres, average="binary")

    # zeros = Yres[YTest == 0]
    # ones = Yres[YTest > 0]
    # data = [zeros, ones]
    # plt.violinplot(data)
    # plt.show()
    
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return area, precision_macro, recall_macro    

def cons_metrics(metrics):
    res = np.array(metrics)
    return np.mean(res, axis = 0).tolist()


def cross_validate(X, Y, folds, classifier_info, **kwargs):
    skf = StratifiedKFold(Y, n_folds=folds)
    metrics = []
    # scalar = StandardScaler()
    for train_ind, test_ind in skf:
        (Yres, Yprobs) = classify(X[train_ind], Y[train_ind], X[test_ind], 
                    classifier_info, **kwargs)
        e = evaluate(Y[test_ind], Yres, Yprobs) 
        metrics.append(e)
    return cons_metrics(metrics)

def mf_cross_validate(train, zeros, folds, mf, K, threshold, shape):
    test_frac = 0.0001
    zeros = zeros.as_matrix()
    metrics = []

    for i in range(folds):
        train_set, test_set = train_test_split(train, test_size = test_frac)
        P, Q = mf(train_set, K, shape = shape)

        test_set = test_set.as_matrix()
        test_length = test_set.shape[0]
        zero_test = sample_cnt(zeros, test_length)
        test_set = np.concatenate((test_set, zero_test), axis = 0)
        test_frame = pd.DataFrame(test_set, columns = ['sender', 'receiver', 'transaction'])    

        (Yres, Yprobs) = MF_predict(P, Q, test_frame, threshold = threshold)

        YTest = test_set[:, 2]
        YTest[YTest > 1] = 1
        

        e = evaluate(YTest, Yres, Yprobs) 
        metrics.append(e)    

    return cons_metrics(metrics)

if __name__ == "__main__":
    train = get_data("txTripletsCounts.txt")
    test = get_data("testTriplets.txt")

    zeros = get_data("zeros.txt")
    shortest_path_train = get_data("shortest_path.txt")
    shortest_path_test = get_data("testing_shortest_path.txt")


    ### Analysis ###
    feature_analysis(train, PF)

    ### Cross Validate ###

    for K in [5, 10, 15, 20, 25]:
        X, Y, _ = get_classifier_X_Y(train, zeros, test, NNegMF, K, K,  
                                            shortest_path_train = None, shortest_path_test = None,
                                            use_blockstate = True)

        print K, metrics_str(cross_validate(X, Y, 10, c_dict['QDA']))

    for K1 in [10, 20]:
        for K2 in [3, 5, 10]:
            X, Y, _ = get_classifier_X_Y(train, zeros, test, NNegMF, K1, K2,  
                                                shortest_path_train = None, shortest_path_test = None,
                                                use_blockstate = True)

            print K1, K2, metrics_str(cross_validate(X, Y, 5, c_dict['QDA']))

    ### Classify ###

    X, Y, XTest = get_classifier_X_Y(train, zeros, test, NNegMF, 10, 10,  
                                        shortest_path_train = None, shortest_path_test = None,
                                        use_blockstate = True)

    (Yres, Yprobs) = classify(X, Y, XTest, c_dict['QDA'])
    YTest = test.as_matrix()[:, 2]
    print metrics_str(evaluate(YTest, Yres, Yprobs))

    ### Cross Validate ###
    shape = (train['sender'].max() + 1, train['receiver'].max() + 1) 

    for K in [3, 5, 10, 20]:
        print K, metrics_str(mf_cross_validate(train, zeros, 5, TSVD, K, 0.0001, shape))    

    for T in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        print T, metrics_str(mf_cross_validate(train, zeros, 5, TSVD, 5, T, shape))

    for K in [3, 5, 10, 20]:
        print K, metrics_str(mf_cross_validate(train, zeros, 5, NNegMF, K, 0.0001, shape))    

    for T in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        print T, metrics_str(mf_cross_validate(train, zeros, 5, NNegMF, 5, T, shape))    

    for T in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        print T, metrics_str(mf_cross_validate(train, zeros, 5, HPF, 2, T, shape))    

    ### Bare Matrix Factorizaion ###

    P, Q = TSVD(train, 5) #0.0001
    # P, Q = NNegMF(train, 5) #0.00001
    # ProbMF(train, 2)
    # P, Q = PF(train, 2)
    # P, Q = HPF(train, 2)

    (Yres, Yprobs) = MF_predict(P, Q, test, threshold = 0.001)
    YTest = test.as_matrix()[:, 2]
    print metrics_str(evaluate(YTest, Yres, Yprobs))
    
