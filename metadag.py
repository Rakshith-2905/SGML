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
        for i in range(FLAGS.num_graph_vertex):
            self.node_cluster_center.append(tf.get_variable(name='{}_{}_node_cluster_center'.format(name, i),
                                                            shape=(1, input_dim)))
        self.vertex_num = FLAGS.num_graph_vertex

        self.GCN = GraphConvolution(self.hidden_dim, name='graph_{}_data_gcn'.format(name))
    
    def compute_metagraph_edges(self,):
        meta_graph = []
        for idx_i in range(self.vertex_num):
            tmp_dist = []
            for idx_j in range(self.vertex_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:

                    # dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                    #     tf.abs(self.node_cluster_center[idx_i] - self.node_cluster_center[idx_j]), units=1,
                    #     name='meta_dist_' + name + '_node_' + str(idx_i)+ '_to_' + str(idx_j))))
                    dist = tf.squeeze(tf.sigmoid(
                        tf.math.reduce_euclidean_norm(self.node_cluster_center[idx_i] - self.node_cluster_center[idx_j]),
                        name='meta_dist'))
                tmp_dist.append(dist)
            meta_graph.append(tf.stack(tmp_dist))
        meta_graph_edges = tf.stack(meta_graph, name='meta_graph_edges')
        return meta_graph_edges

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

        self.meta_graph_edges = self.compute_metagraph_edges()

        # print("meta_graph ", meta_graph)
        proto_graph = []
        for idx_i in range(self.proto_num):
            tmp_dist = []
            for idx_j in range(self.proto_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(
                        tf.math.reduce_euclidean_norm(inputs[idx_i] - inputs[idx_j]),name='proto_dist'))
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
            
    def graph_embedding(self, graph_nodes, inp_graph_edges, level_num, graph_num):
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

        if inp_graph_edges == None:
            # Compute the edges/affinity between nodes of the graph to be embedded
            graph_connections = []
            for idx_i in range(FLAGS.num_classes):
                tmp_dist = []
                for idx_j in range(FLAGS.num_classes):
                    if idx_i == idx_j:
                        dist = tf.squeeze(tf.zeros([1]))
                    else:                    
                        dist = tf.squeeze(tf.sigmoid(
                            tf.math.reduce_euclidean_norm(graph_nodes[idx_i] - graph_nodes[idx_j]),name='proto_dist'))
                    tmp_dist.append(dist)
                graph_connections.append(tf.stack(tmp_dist))

            graph_edges = tf.stack(graph_connections, name='proto_l{}_g{}_edges'.format(level_num, graph_num))
        else:
            graph_edges = inp_graph_edges
        # feature matrix is obtained as the product of degree-normalized adjacency matrix and graph nodes
        if not self.eigen_embedding:
            node_size = tf.shape(graph_edges)[0]
            I = tf.eye(node_size)
            adj = graph_edges + I
            D = tf.diag(tf.reduce_sum(adj, axis=1))
            adj = tf.matmul(tf.linalg.inv(D), adj)
            # Normalizing by degree sum(adj, axis=0) is degree of that row
            # adj = adj / tf.reduce_sum(adj, axis=0)
            feature_matrix = tf.matmul(adj, graph_nodes)
        else:
            # Kroneckers product of the eigen vectors of affinity matrix and the graph nodes
            eigen_vectors = tf.linalg.eigh(graph_edges, name='level_{}_graph_{}_eigen_vectors'.format(level_num, graph_num))[1]
            feature_matrix = tf_kron(graph_nodes, eigen_vectors)

        # Mean of the feature matrix
        embedding = tf.math.reduce_mean(feature_matrix, axis=0)
        
        if inp_graph_edges == None:
            feature_matrix = tf.identity(feature_matrix, name='level_{}_graph_{}_proto_GCN_feature'.format(level_num, graph_num))
            embedding = tf.identity(embedding, name='level_{}_graph_{}_embedding'.format(level_num, graph_num))
        
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
        sigma = 2.0
        tree_graphs = []
        tree_embeddings = []
        # Iterate though each level of the graph tree
        for i, level in enumerate(self.graph_tree):
            updated_proto_graphs = []
            updated_proto_embeddings = []
            # If first level compute the updated proto graph as GCN(proto graph, current meta graph)
            if i == 0:
                updated_proto_graph_nodes, updated_proto_graph_edges = level[0].model(inputs)

                updated_proto_graphs.append(updated_proto_graph_nodes)

                updated_proto_embeddings.append(self.graph_embedding(updated_proto_graph_nodes, updated_proto_graph_edges, i, 0))
            else:
                # Iterate over the updated prototype graph and embeddings from the previous level
                for j, prev_level_updated_graph, prev_level_updated_embedding in zip(range(len(tree_embeddings[-1])), tree_graphs[-1], tree_embeddings[-1]):

                    # Compute the soft assignment between the current level meta graphs and the updated prototype graph
                    soft_attention = []
                    for k, graph in enumerate(level):
                        # Compute the embedding for the current graph
                        current_graph = tf.squeeze(tf.stack(graph.node_cluster_center), axis=1)
                        current_embedding = self.graph_embedding(current_graph, graph.compute_metagraph_edges(), i, k) 
                        euclid_diff = tf.reduce_sum(tf.square(current_embedding - prev_level_updated_embedding
                        ), name="level_{}_graph_{}_level_{}_graph_{}_euclid_diff".format(i-1,j,i,k))  
                        soft_attention.append(tf.exp(-euclid_diff/ (2.0 * sigma)))
                    # Do an sigmoid operation on the attentions
                    soft_attention = tf.stack(soft_attention)/tf.reduce_sum(tf.stack(soft_attention))

                    # # equal attention
                    # soft_attention = tf.ones(len(level))
                    # soft_attention = soft_attention/tf.reduce_sum(soft_attention)
                
                    soft_attention = tf.identity(soft_attention, name='attention_l{}n{}_to_l{}'.format(i-1, j, i))
                    # Iterate through the current level meta graphs
                    temp_updated_graphs = []
                    for k, graph in enumerate(level):
                        # compute the updated proto graph as GCN(updated proto graph, current meta graph)
                        graph_nodes,_  = graph.model(prev_level_updated_graph)
                        # Store the current updated_proto_graphs weighted by the attention
                        temp_updated_graphs.append(soft_attention[k] * graph_nodes)
                    updated_proto_graphs.append(tf.stack(temp_updated_graphs))

                # Sum the multiple sets of updated prototype graph generated by each previous level prototype graph
                updated_proto_graphs = tf.stack(updated_proto_graphs)
                updated_proto_graphs = tf.reduce_sum(updated_proto_graphs, axis=0, keepdims=False)
                updated_proto_graphs = tf.unstack(updated_proto_graphs, axis=0)
                # Compute the embedding for the updated prototype graphs of the current level
                for k, graph in enumerate(updated_proto_graphs):
                    updated_proto_embeddings.append(self.graph_embedding(graph, inp_graph_edges=None, level_num=i, graph_num=k))

            # Update the history of updated prototype graph and embeddings list 
            tree_graphs.append(updated_proto_graphs)
            tree_embeddings.append(updated_proto_embeddings)

            # print("\n***********************************************\n\n\n")
        return tree_graphs[-1][0]

if __name__ == "__main__":
    main()
