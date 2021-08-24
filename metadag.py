from abc import abstractmethod
import ipdb
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import tf_kron
FLAGS = flags.FLAGS

class GraphAttentionNetwork(object):

    def __init__(self, hidden_dim, name=None, num_head=1, sparse_inputs=False, act=tf.nn.tanh, bias=True, dropout=0.0):
        self.act = act
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.num_heads = num_head

        with tf.variable_scope('{}_vars'.format(name), tf.AUTO_REUSE):
            self.gcn_weights = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], dtype=tf.float32),
                                           name='gcn_weight')
            self.att_weights = []
            for i in range(num_head):
                self.att_weights.append(tf.Variable(tf.truncated_normal([2*self.hidden_dim, 1], dtype=tf.float32),
                                            name='att_weight_' + str(i)))

    def model(self, feat, adj):
        h = feat
        h = tf.nn.dropout(h, 1 - self.dropout)

        node_size = tf.shape(adj)[0]
        I = tf.eye(node_size)
        adj = adj + I

        num_meta_vertex = adj.shape[1] - FLAGS.num_classes
        num_meta_graphs = num_meta_vertex//FLAGS.num_graph_vertex
        
        # multiply the feature vector by the weight h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = tf.matmul(h, self.gcn_weights)
        
        # multi-head self-attention
        h_primes = []
        l1_att = []       
        l2_att = []                                  
        for i in range(self.num_heads):
            # Wh1&2.shape (N, 1)
            Wh1 = tf.matmul(Wh, self.att_weights[i][:self.hidden_dim, :])
            Wh2 = tf.matmul(Wh, self.att_weights[i][self.hidden_dim:, :])

            # e.shape (N, N)
            e = Wh1 + tf.transpose(Wh2)
            e = tf.nn.leaky_relu(e)

            zero_vec = -9e15*tf.zeros(adj.shape)
            alpha = tf.where(adj > 0, e, zero_vec)

            alpha = tf.nn.softmax(alpha, axis = 1)

            # L1, L2 regularization on each meta node
            alpha_reduced = alpha[:FLAGS.num_classes, FLAGS.num_classes:]
            l1 = tf.reduce_sum(tf.math.abs(alpha_reduced))
            l1_att.append(l1)
            for j in range(num_meta_graphs):
                meta_alpha = alpha_reduced[:, FLAGS.num_graph_vertex*j:FLAGS.num_graph_vertex*(j+1)]
                l2_meta = tf.math.sqrt(tf.reduce_sum(tf.math.square(meta_alpha)))
                l2_meta = tf.math.sqrt(tf.reduce_sum(tf.math.square(alpha), axis=0))
                l2_att.append(tf.reduce_sum(l2_meta))

            # Get the updated feature representation
            h_prime = tf.matmul(alpha, Wh)
            h_primes.append(h_prime)

        output = tf.reduce_mean(tf.stack(h_primes), axis=0)

        l1 = tf.reduce_sum(tf.stack(l1_att))
        l2 = tf.reduce_sum(tf.stack(l2_att))

        if self.act is not None:
            return self.act(output), l1, l2
        else:
            return output[0:self.proto_num], l1, l2


# class GraphAttentionNetwork(object):

#     def __init__(self, hidden_dim, name=None, num_head=1, sparse_inputs=False, act=tf.nn.tanh, bias=True, dropout=0.0):
#         self.act = act
#         self.dropout = dropout
#         self.sparse_inputs = sparse_inputs
#         self.hidden_dim = hidden_dim
#         self.bias = bias
#         self.num_heads = num_head

#         # Create weights for each attention head
#         # Create weights for the GCN
#         with tf.variable_scope('{}_vars'.format(name), tf.AUTO_REUSE):
#             self.gcn_weights = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], dtype=tf.float32),
#                                            name='gcn_weight')
#             self.att_weights = []
#             for i in range(num_head):
#                 self.att_weights.append(tf.Variable(tf.truncated_normal([2*self.hidden_dim, 1], dtype=tf.float32),
#                                             name='att_weight_' + str(i)))

#     def model(self, prototype, meta_graphs, super_edges):

#         # multiply the feature vector by the weight h.shape: (N, in_features), Wh.shape: (N, out_features)
#         W_proto = tf.matmul(prototype, self.gcn_weights)
#         W_meta = tf.matmul(meta_graphs, self.gcn_weights)
#         # Compute the neighbour map and concatinate the neighbours
#         num_super_nodes = super_edges.shape[1]
        
#         l1_att = []       
#         l2_att = []
#         # multi-head self-attention
#         multi_head_updated_proto = []           
#         for i in range(self.num_heads):
#             updated_proto_graph = []
#             for j in range(FLAGS.num_classes):
#                 alpha_meta = []
#                 for k in range(num_super_nodes):
#                     e = tf.matmul(tf.expand_dims(tf.concat((W_proto[j], W_meta[k]), axis=0), axis=0), self.att_weights[i])
#                     e = tf.nn.leaky_relu(e)
#                     alpha = tf.where(super_edges[j][k]>0,  tf.squeeze(e), -9e15)
#                     alpha_meta.append(alpha)
                
#                 # Softmax of attention
#                 alpha_meta = tf.nn.softmax(tf.stack(alpha_meta))
#                 temp_updated_node = []
#                 for k in range(num_super_nodes):
#                     temp_updated_node.append((alpha_meta[k] * W_meta[k][:]))
#                 updated_proto_node = tf.sigmoid(tf.reduce_sum(temp_updated_node, axis=0))
#                 updated_proto_graph.append(updated_proto_node)

#                 # Reshape the attention as num_meta_graphs, num_meta_nodes.
#                 alpha_meta = tf.reshape(alpha_meta, shape=[num_super_nodes//FLAGS.num_graph_vertex, FLAGS.num_graph_vertex])

#                 # L1 reg of attention across all metagraphs
#                 l1 = tf.reduce_sum(tf.math.abs(alpha_meta))
#                 # L2 of attention of each of the metagraph
#                 l2 = tf.math.sqrt(tf.reduce_sum(tf.math.square(alpha_meta), axis=1))
#                 l1_att.append(l1)
#                 l2_att.append(tf.reduce_sum(l2))

#             multi_head_updated_proto.append(tf.stack(updated_proto_graph))
            
#         output = tf.reduce_mean(tf.stack(multi_head_updated_proto), axis=0)

#         l1 = tf.reduce_sum(tf.stack(l1_att))
#         l2 = tf.reduce_sum(tf.stack(l2_att))

#         if self.act is not None:
#             return self.act(output), 0, 0
#         else:
#             return output, l1, l2

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
            self.GCN.append(GraphAttentionNetwork(self.hidden_dim, num_head = FLAGS.att_head, name='level_{}_data_gcn'.format(i)))
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

        l1_regularization = []
        l2_regularization = []
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
            print(proto_edges)
            print(super_edges)
            # merge the supergraph and the prototype graph as Ap, As, As_transpose, Am
            adj = tf.concat((tf.concat((proto_edges, super_edges), axis=1),
                            tf.concat((tf.transpose(super_edges, perm=[1, 0]), meta_edges), axis=1)), axis=0)
            feat = tf.concat((inputs, meta_features), axis=0)

            inputs, l1, l2 = self.GCN[i].model(feat, adj)
            
            l1_regularization.append(l1)
            l2_regularization.append(l2)
        
        l1_regularization = tf.reduce_sum(tf.stack(l1_regularization))
        l2_regularization = tf.reduce_sum(tf.stack(l2_regularization))

        return inputs, l1_regularization, l2_regularization

if __name__ == "__main__":
    main()
