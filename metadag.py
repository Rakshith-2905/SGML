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
        l1_l2_att = []                                  
        for i in range(self.num_heads):
            # Wh1&2, shape:(N, 1)
            Wh1 = tf.matmul(Wh, self.att_weights[i][:self.hidden_dim, :])
            Wh2 = tf.matmul(Wh, self.att_weights[i][self.hidden_dim:, :])

            # e, shape (N, N)
            e = Wh1 + tf.transpose(Wh2)
            e = tf.nn.leaky_relu(e)
            
            # If the edge doesnt exist between nodes append zero else use the relationship between the node features
            zero_vec = -9e15*tf.zeros(adj.shape)
            alpha = tf.where(adj > 0, e, zero_vec)

            # attention matrix, shape:(num_supernodes,num_supernodes)
            alpha = tf.nn.softmax(alpha, axis = 1)

            # extract the proto type and meta nodes realtionship, shape:(num_proto, num_metanodes)(each metanodes has a meta graph)
            alpha_reduced = alpha[:FLAGS.num_classes, FLAGS.num_classes:]

            if FLAGS.proto_group_sparsity:
                l2_proto_meta_nodes = []
                for j in range(FLAGS.num_classes):
                    l2_meta_nodes = []
                    # Iterate over each meta graph
                    for k in range(num_meta_graphs):
                        # extract the current meta graph, shape:(num_meta_vertex, num_meta_vertex)
                        meta_alpha = alpha_reduced[j, FLAGS.num_graph_vertex*k:FLAGS.num_graph_vertex*(k+1)]
                        # Compute l2 norm for the current meta graph and proto graph node alpha
                        l2_meta = tf.math.sqrt(tf.reduce_sum(tf.math.square(meta_alpha)))
                        l2_meta_nodes.append(l2_meta)

                    # Compute the l1 of the l2 of each meta nodes
                    l2_proto_meta_nodes.append(tf.reduce_sum(tf.stack(l2_meta_nodes)))
                # Sum the group sparsity of each proto graph node
                l1_l2_att.append(tf.reduce_sum(tf.stack(l2_proto_meta_nodes)))

            else:
                l2_meta_nodes = []
                # Iterate over each meta graph
                for j in range(num_meta_graphs):
                    # extract the current meta graph, shape:(num_meta_vertex, num_meta_vertex)
                    meta_alpha = alpha_reduced[:, FLAGS.num_graph_vertex*j:FLAGS.num_graph_vertex*(j+1)]
                    # Compute l2 norm for the current meta graph
                    l2_meta = tf.math.sqrt(tf.reduce_sum(tf.math.square(meta_alpha)))
                    l2_meta_nodes.append(l2_meta)

                # Compute the l1 of the l2 of each meta nodes
                l1_l2_att.append(tf.reduce_sum(tf.stack(l2_meta_nodes)))

            
            # Get the updated feature representation
            h_prime = tf.matmul(alpha, Wh)
            h_primes.append(h_prime)

        # Sum the updated feature representation for each attention head
        output = tf.reduce_mean(tf.stack(h_primes), axis=0)

        # Sum the l1_l2_norm for each attention head
        group_sparsity = tf.reduce_sum(tf.stack(l1_l2_att))

        if self.act is not None:
            return self.act(output), group_sparsity
        else:
            return output, group_sparsity

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

        group_sparsity_tree = []
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

            inputs, group_sparsity = self.GCN[i].model(feat, adj)
            inputs = inputs[0:FLAGS.num_classes]

            group_sparsity_tree.append(group_sparsity)
        
        group_sparsity = tf.reduce_sum(tf.stack(group_sparsity_tree))

        return inputs, group_sparsity

if __name__ == "__main__":
    main()
