from graph_tool.all import * 
import numpy as np
import pandas as pd

DATA_PATH = "../data/%s"

def get_train_data(fname):
    return pd.read_csv(DATA_PATH % fname,
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])

def load_graph(train, graph_file_name):
    g = Graph(directed=True)
    address_to_vertex = {}
    edgeweight_double = g.new_edge_property("int")

    #Should we include vertices that have no transactions?
    #Iterate through all addresses that were used
    for sender in train['sender']:
        if sender not in address_to_vertex:
            v = g.add_vertex()
            address_to_vertex[sender] = v

    for receiver in train['receiver']:
        if receiver not in address_to_vertex:
            v = g.add_vertex()
            address_to_vertex[receiver] = v
    
    #Draw the Edges
    for transaction_index in range(0,len(train['sender'])):
        e = g.add_edge(address_to_vertex[train['sender'][transaction_index]], 
                       address_to_vertex[train['receiver'][transaction_index]])

        #Set the weight of the edge
        edgeweight_double[e] = train['transaction'][transaction_index]

    g.save(graph_file_name)

def graph_inference(file_name):
    g = graph_tool.load_graph(file_name)
#   block_state = minimize_blockmodel_dl(g, overlap=True)
    state = BlockState(g, B=276, deg_corr = True)
    pv = None
    state.mcmc_sweep(niter=1000)
    for i in range(1000):
        ds, nmoves = state.mcmc_sweep(niter=10)
        pv = state.collect_vertex_marginals(pv)
    print mf_entropy(g, pv)

    graph_draw(g)
    
if __name__ == "__main__":
    graph_file = "bitcoin_graph.xml.gz"
    train = get_train_data("txTripletsCounts.txt")
    #load_graph(train, graph_file)
    graph_inference(graph_file)
