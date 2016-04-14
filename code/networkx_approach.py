import networkx as nx
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "../data/%s"

def get_train_data(fname):
    return pd.read_csv(DATA_PATH % fname,
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])

def load_graph(train):
    print 'about to load the graph'
    g = nx.DiGraph()

    for index in range(0,len(train['sender'])):
        g.add_edge(train['sender'][index], train['receiver'][index], weight=train['transaction'][index])

    print 'graph has been loaded'
    return g


def graph_inference(g):
    print 'starting the graph inference'
    path_length_dictionary = nx.shortest_path_length(g)
    print path_length_dictionary
    
    
if __name__ == "__main__":
    train = get_train_data("txTripletsCounts.txt")
    graph = load_graph(train)
    graph_inference(graph)
