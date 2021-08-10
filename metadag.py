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
        self.name = name
        self.node_cluster_center, self.nodes_cluster_bias = [], []
        self.w, self.v, self.u = [], [], []
        for i in range(FLAGS.num_graph_vertex):
            self.node_cluster_center.append(tf.get_variable(name='{}_{}_node_cluster_center'.format(name, i),
                                                            shape=(1, input_dim)))
        
            self.w.append(tf.get_variable(name='{}_{}_w'.format(name,i),
                                                            shape=(FLAGS.num_classes, 1)))

            self.v.append(tf.get_variable(name='{}_{}_v'.format(name,i),
                                                            shape=(FLAGS.num_classes, input_dim)))

            self.u.append(tf.get_variable(name='{}_{}_u'.format(name,i),
                                                            shape=(FLAGS.num_classes, input_dim)))
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

        self.GCN = GraphConvolution(self.hidden_dim, name='graph_{}_data_gcn'.format(name))

    def model(self, inputs):
        """
        Computes proto_graph, Meta graph, and super_graph
        pass message between the super_graph and proto_graph using GNN

        Parameters
        ----------
        inputs : tensor, shape: num_class, embedding_dims
        
        Return
        ------
        repr : Tensor, shape: num_class, 128
        """
        # print("MetaGraph Input: ", inputs)
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            sigma = 8.0
        elif FLAGS.datasource in ['2D']:
            sigma = 2.0

        cross_graph = tf.nn.softmax(
            (-tf.reduce_sum(tf.square(inputs - self.node_cluster_center), axis=-1) / (2.0 * sigma)), axis=0)
        cross_graph = tf.transpose(cross_graph, perm=[1, 0])
        # print("cross_graph ", cross_graph)

        # print("meta_graph ", meta_graph)
        proto_graph = []
        for idx_i in range(self.proto_num):
            tmp_dist = []
            for idx_j in range(self.proto_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(tf.expand_dims(inputs[idx_i] - inputs[idx_j], axis=0)), units=1, name='proto_dist')))
                tmp_dist.append(dist)
            proto_graph.append(tf.stack(tmp_dist))
        proto_graph = tf.stack(proto_graph)
        # print("proto_graph ", proto_graph)

        adj = tf.concat((tf.concat((proto_graph, cross_graph), axis=1),
                         tf.concat((tf.transpose(cross_graph, perm=[1, 0]), self.meta_graph_edges), axis=1)), axis=0)

        feat = tf.concat((inputs, tf.squeeze(tf.stack(self.node_cluster_center))), axis=0)

        repr = self.GCN.model(feat, adj)
        repr = repr[0:self.proto_num]
        repr = tf.identity(repr, name=self.name + '_updated')
        return repr, proto_graph


class TreeGraph(object):
    def __init__(self, hidden_dim, input_dim):
        self.eigen_embedding = FLAGS.eigen_embedding
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize the meta graphs for each level and store it in self.graph_tree
        self.graph_tree = []
        for i, num_graph in enumerate(FLAGS.graph_list):
            level_list = []
            for j in range(num_graph):
                level_list.append(MetaGraph(input_dim=FLAGS.hidden_dim, hidden_dim=FLAGS.hidden_dim, name='level_{}_graph_{}'.format(str(i), str(j))))
            self.graph_tree.append(level_list)
            
    def graph_embedding(self, graph_nodes, graph_edges, level_num, graph_num):
        """
        Computes an embedding for the given graph nodes. 

        Parameters
        ----------
        graph_nodes : tensor, shape: [num_class, embedding_dims] Represents the graph to be embedded

        graph_edges : tensor, shape: [num_class, num_class] Represents edge values between the nodes of a graph
        
        Return
        ------
        embedding : Tensor, shape: [1, embedding_dims]
        """
        if graph_edges == None:
            # Compute the edges/affinity between nodes of the graph to be embedded
            graph_connections = []
            for idx_i in range(FLAGS.num_classes):
                tmp_dist = []
                for idx_j in range(FLAGS.num_classes):
                    if idx_i == idx_j:
                        dist = tf.squeeze(tf.zeros([1]))
                    else:
                        dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                            tf.abs(tf.expand_dims(graph_nodes[idx_i] - graph_nodes[idx_j], axis=0)), units=1, name='proto_dist')))
                    tmp_dist.append(dist)
                graph_connections.append(tf.stack(tmp_dist))

            graph_edges = tf.stack(graph_connections)

        # feature matrix is obtained as the product of degree-normalized adjacency matrix and graph nodes
        if not self.eigen_embedding:
            node_size = tf.shape(graph_edges)[0]
            I = tf.eye(node_size)
            adj = graph_edges + I
            D = tf.diag(tf.reduce_sum(adj, axis=1))
            adj = tf.matmul(tf.linalg.inv(D), adj)
            feature_matrix = tf.matmul(adj, graph_nodes)
        else:
            # Kroneckers product of the eigen vectors of affinity matrix and the graph nodes
            eigen_vectors = tf.linalg.eigh(graph_edges, name='level_{}_graph_{}_eigen_vectors'.format(level_num, graph_num))[1]
            feature_matrix = tf_kron(graph_nodes, eigen_vectors)

        # Mean of the feature matrix
        embedding = tf.math.reduce_mean(feature_matrix, axis=0, name='level_{}_graph_{}_embedding'.format(level_num, graph_num))
        return  embedding

    def model(self, inputs):
        """
        Computes proto_graph, Meta graph, and super_graph
        pass message between the super_graph and proto_graph using GNN

        Parameters
        ----------
        inputs : tensor, shape: [num_class, embedding_dims] Represents the proto graph nodes
        
        Return
        ------
        repr : Tensor, shape: [num_class, 128]
        """
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            sigma = 8.0
        elif FLAGS.datasource in ['2D']:
            sigma = 2.0

        # Iterate though each level of the graph tree
        for i, level in enumerate(self.graph_tree):

            # Iterate through the current level meta graphs and compute the updated proto graph
            soft_attention = []
            temp_updated_graphs = []
            for graph in level:
                # compute the updated proto graph as GCN(updated proto graph, current meta graph)
                graph_nodes,_  = graph.model(inputs)

                node_att =[]
                for j in range(FLAGS.num_graph_vertex):
                    node = tf.expand_dims(graph_nodes[j], axis=0)
                    v_xt = tf.tanh(tf.matmul(graph.v[j],tf.transpose(node)))
                    u_xt = tf.sigmoid(tf.matmul(graph.u[j],tf.transpose(node)))
                    tensor_pro = tf.multiply(v_xt, u_xt)
                    node_att.append(tf.transpose(tf.expand_dims(graph.w[j], axis=1))@tensor_pro)
                
                node_att = tf.squeeze(tf.stack(node_att))
                soft_attention.append(node_att)
                # Store the current updated_proto_graphs weighted by the attention
                temp_updated_graphs.append(graph_nodes)
            
            temp_updated_graphs = tf.stack(temp_updated_graphs)
            soft_attention = tf.stack(soft_attention)
            # Compute the softmax for every node of every graph in the level
            soft_attention = tf.nn.softmax(soft_attention, axis=0, name='attention_l{}'.format(i))
            
            updated_graphs = []
            for i in range(len(level)):
                updated_nodes = []
            # Update each node as the weighted average
                for j in range(FLAGS.num_graph_vertex):
                    updated_nodes.append(soft_attention[i][j]*temp_updated_graphs[i][j])
                updated_graphs.append(tf.stack(updated_nodes))
            updated_graphs = tf.stack(updated_graphs)

            inputs = tf.reduce_sum(updated_graphs, axis = 0)
            # print("\n***********************************************\n\n\n")
        return inputs

if __name__ == "__main__":
    main()
