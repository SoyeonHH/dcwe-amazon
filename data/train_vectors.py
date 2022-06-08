import argparse
import pickle

import networkx as nx
from node2vec import Node2Vec


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=0, required=True, type=int, help='Size of embeddings.')
    args = parser.parse_args()

    with open('amazon_fashion_edges.p', 'rb') as f:
        edge_set = pickle.load(f)
    with open('amazon_fashion_products.p', 'rb') as f:
        users = pickle.load(f)

    graph = nx.DiGraph()
    graph.add_nodes_from(users)
    graph.add_edges_from(edge_set)

    assert graph.number_of_nodes() == len(users)
    assert graph.number_of_edges() == len(edge_set)

    n2v = Node2Vec(graph, dimensions=args.dim, walk_length=80, num_walks=10, workers=1)
    n2v_model = n2v.fit(window=2, min_count=1, seed=123)   # omit iter=10

    n2v_model.wv.save_word2vec_format('amazon_fashion_vectors_{}.txt'.format(args.dim), binary=False)


if __name__ == '__main__':
    main()
