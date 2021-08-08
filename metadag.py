from abc import abstractmethod
import ipdb
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import tf_kron
FLAGS = flags.FLAGS

class GraphConvolution(object):

    def __init__(self, hidden_dim, name=None, sparse_inputs=False, act=tf.nn.tanh, bias=True, dropout=0.0):
        self.act = act
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias

        with tf.variable_scope('{}_vars'.format(name), tf.AUTO_REUSE):
            self.gcn_weights = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], dtype=tf.float32),
                                           name='gcn_weight')
            if self.bias:
                self.gcn_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_dim],
                                                        dtype=tf.float32), name='gcn_bias')

    def model(self, feat, adj):
        x = feat
        x = tf.nn.dropout(x, 1 - self.dropout)

        node_size = tf.shape(adj)[0]
        I = tf.eye(node_size)
        adj = adj + I
        D = tf.diag(tf.reduce_sum(adj, axis=1))
        adj = tf.matmul(tf.linalg.inv(D), adj)
        pre_sup = tf.matmul(x, self.gcn_weights)
        output = tf.matmul(adj, pre_sup)
        if self.bias:
            output += self.gcn_bias
        if self.act is not None:
            return self.act(output)
        else:
            return output

class MetaGraph(object):
    def __init__(self, hidden_dim, input_dim, name=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proto_num = FLAGS.num_classes
        self.node_cluster_center, self.nodes_cluster_bias = [], []

        for i in range(FLAGS.num_graph_vertex):
            self.node_cluster_center.append(tf.get_variable(name='graph_{}_{}_node_cluster_center'.format(name, i),
                                                            shape=(1, input_dim)))
            self.nodes_cluster_bias.append(
                tf.get_variable(name='graph_{}_{}_nodes_cluster_bias'.format(name, i), shape=(1, hidden_dim)))

        self.vertex_num = FLAGS.num_graph_vertex


        meta_graph = []
        for idx_i in range(self.vertex_num):
            tmp_dist = []
            for idx_j in range(self.vertex_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(self.node_cluster_center[idx_i] - self.node_cluster_center[idx_j]), units=1,
                        name='meta_dist_' + name + '_node_' + str(idx_i)+ '_to_' + str(idx_j))))
                tmp_dist.append(dist)
            meta_graph.append(tf.stack(tmp_dist))
        self.meta_graph_edges = tf.stack(meta_graph, name='meta_graph_edges_' + name)


class TreeGraph(object):
    def __init__(self, hidden_dim, input_dim):
        self.eigen_embedding = FLAGS.eigen_embedding
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize the meta graphs for each level and store it in self.graph_tree
        self.graph_tree = []
        self.GCN = []
        for i, num_graph in enumerate(FLAGS.graph_list):
            level_list = []
            self.GCN.append(GraphConvolution(self.hidden_dim, name='level_{}_data_gcn'.format(i)))
            for j in range(num_graph):
                level_list.append(MetaGraph(input_dim=FLAGS.hidden_dim, hidden_dim=FLAGS.hidden_dim, name='level_{}_graph_{}'.format(str(i), str(j))))
            self.graph_tree.append(level_list)
    
    def model(self, inputs):
        """
        Computes proto_graph, Meta graph, and super_graph
        pass message between the super_graph and proto_graph using GNN

        Parameters
        ----------
        inputs : tensor, shape: [num_class, embedding_dims] Represents the proto graph nodes
        
        Return
        ------
        inputs : Tensor, shape: [num_class, 128] Knowledge graph updated proto graph nodes
        """
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            sigma = 8.0
        elif FLAGS.datasource in ['2D']:
            sigma = 2.0

        # print("input : ", inputs.shape)
        # Iterate though each level of the graph tree
        for i, level in enumerate(self.graph_tree):
            #  Compute the prtotype edges
            proto_edges = []
            for idx_i in range(FLAGS.num_classes):
                tmp_dist = []
                for idx_j in range(FLAGS.num_classes):
                    if idx_i == idx_j:
                        dist = tf.squeeze(tf.zeros([1]))
                    else:
                        dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                            tf.abs(tf.expand_dims(inputs[idx_i] - inputs[idx_j], axis=0)), units=1, name='proto_dist')))
                    tmp_dist.append(dist)
                proto_edges.append(tf.stack(tmp_dist))
            proto_edges = tf.stack(proto_edges)

            # Iterate through each graph on the level
            meta_features = []
            meta_edges = []
            super_edges = []
            for j, graph in enumerate(level):
                meta_features.append(tf.squeeze(tf.stack(graph.node_cluster_center)))
                # Edges of zero between the current metagraph and other meta graphs in the level
                meta_edges.append(tf.concat((graph.meta_graph_edges, tf.zeros(
                    (FLAGS.num_graph_vertex, (len(level)-1)*FLAGS.num_graph_vertex))),axis=1))
                
                # Edge between the current meta graph and prototype graph, softmax(l2 dist of nodes)
                cross_graph = tf.nn.softmax(
                    (-tf.reduce_sum(tf.square(inputs - graph.node_cluster_center), axis=-1) / (2.0 * sigma)), axis=0)
                super_edges.append(tf.transpose(cross_graph, perm=[1, 0]))
            
            meta_features = tf.concat(meta_features, axis=0)
            meta_edges = tf.concat(meta_edges, axis=0)
            super_edges = tf.concat(super_edges, axis=1)
            
            # merge the supergraph and the prototype graph as Ap, As, As_transpose, Am
            adj = tf.concat((tf.concat((proto_edges, super_edges), axis=1),
                            tf.concat((tf.transpose(super_edges, perm=[1, 0]), meta_edges), axis=1)), axis=0)
            feat = tf.concat((inputs, meta_features), axis=0)

            inputs = self.GCN[i].model(feat, adj)[0:FLAGS.num_classes]
            
            # print("level: {} number of graphs: {} num of vertex: {}".format(j,len(level), FLAGS.num_graph_vertex))
            # print("proto edges: ", proto_edges.shape)
            # print("meta edges: ", meta_edges.shape)
            # print("meta features: ", meta_features.shape)
            # print("super edges: ", super_edges.shape)
            # print("adj : ", adj.shape)
            # print("feat : ", feat.shape)
            # print("updated input : ", inputs.shape)

            # print("\n***********************************************\n\n\n")
        return inputs

if __name__ == "__main__":
    main()
