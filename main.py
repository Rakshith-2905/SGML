import random
import time
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.random.set_random_seed(1234)
from maml import MAML

tf.set_random_seed(1234)
from data_generator import DataGenerator
from tensorflow.python.platform import flags
from scipy.spatial import distance
from scipy.special import softmax

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'plainmulti', '2D or plainmulti or artmulti')
flags.DEFINE_bool('hetrogeneous', False, 'Sample data hetrogenously across 4 classes')
flags.DEFINE_integer('test_dataset', -1,
                     'which data to be test, plainmulti: 0-3, artmulti: 0-11, -1: random select')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_test_task', 1000, 'number of test tasks.')
flags.DEFINE_integer('test_epoch', 0, 'test epoch, only work when test start')

## Training options
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('update_batch_size_eval', 10,
                     'number of examples used for inner gradient test (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('num_updates_test', 20, 'number of inner gradient updates during training.')
flags.DEFINE_integer('sync_group_num', 6, 'the number of different groups in sync dataset')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, or None')
flags.DEFINE_integer('hidden_dim', 40, 'output dimension of task embedding')
flags.DEFINE_integer('num_filters', 64, '32 for plainmulti and artmulti')
flags.DEFINE_integer('sync_filters', 40, 'number of dim when combine sync functions.')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('emb_loss_weight', 0.0, 'the weight of autoencoder')
flags.DEFINE_integer('task_embedding_num_filters', 32, 'number of filters for task embedding')
## Graph information
flags.DEFINE_list('graph_list', [1,2,1], 'list of nodes in each level')
flags.DEFINE_integer('num_graph_vertex', 4, 'number of vertex for each of the graphs in all the layers')
flags.DEFINE_bool('eigen_embedding', False, 'Method for embedding the meta graph')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './logs/', 'directory for summaries and checkpoints.')
flags.DEFINE_string('datadir', './Data/', 'directory for datasets.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', True, 'Set to true to evaluate on the the test set, False for the validation set.')

flags.DEFINE_bool('DEBUG', False, 'Print the difference between the graphs of diff levels')


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SAVE_INTERVAL = 10000
    if FLAGS.datasource in ['2D']:
        PRINT_INTERVAL = 1000
    else:
        PRINT_INTERVAL = 100

    print('Done initializing, starting training.')

    meta_train_acc, meta_test_acc, meta_train_loss, combined_loss = [], [], [], []

    attention =[]
    meta_graph, meta_graph_edges, graph_embeddings, embed_diffs = [], [], [], []

    num_classes = data_generator.num_classes
    start_time = time.time()
    for itr in range(resume_itr, FLAGS.metatrain_iterations):

        feed_dict = {}
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

        if itr == 0:
            input_tensors = [model.combined_loss, model.combined_loss, model.total_embed_loss, model.total_loss1,
                         model.total_losses2[FLAGS.num_updates - 1]]
        else:
            input_tensors = [model.metatrain_op, model.combined_loss, model.total_embed_loss, model.total_loss1,
                            model.total_losses2[FLAGS.num_updates - 1]]
        if model.classification:
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

            attention_fetch = []
            for level, graphs in enumerate(FLAGS.graph_list[:-1]):
                attention_level = []
                for graph_idx in range(graphs):
                    attention_level.append('model/attention_l{}n{}_to_l{}:0'.format(level, graph_idx, level+1))
                attention_fetch.append(attention_level)
            input_tensors.append(attention_fetch)

            tree_graphs = []
            tree_graph_edges = []
            for level, num_graphs in enumerate(FLAGS.graph_list):
                graph_list = []
                graph_edges = []
                
                for graph_idx in range(num_graphs):
                    node_list = []
                    graph_edges.append('meta_graph_edges_level_{}_graph_{}:0'.format(level, graph_idx))
                    
                    for node in range(FLAGS.num_graph_vertex):
                        node_list.append('level_{}_graph_{}_{}_node_cluster_center:0'.format(level, graph_idx, node))
                    graph_list.append(node_list)

                tree_graphs.append(graph_list)
                tree_graph_edges.append(graph_edges)
            
            
            embedding_fetch = []
            for level, graphs in enumerate(FLAGS.graph_list):
                embedding_level = []
                for graph_idx in range(graphs):
                        if level == 0:
                            embedding_level.append('model/level_{}_graph_{}_embedding:0'.format(level, graph_idx))
                            pass
                        else:
                            embedding_level.append('model/level_{}_graph_{}_embedding_1:0'.format(level, graph_idx))
                embedding_fetch.append(embedding_level)

            diff_fetch = []
            for prev_level, prev_graphs in enumerate(FLAGS.graph_list[:-1]):
                diff_level = []
                for prev_graph_idx in range(prev_graphs):
                        for curr_graph_index in range(FLAGS.graph_list[prev_level+1]):
                            diff_level.append('model/level_{}_graph_{}_level_{}_graph_{}_euclid_diff:0'.format(
                                prev_level, prev_graph_idx, prev_level+1, curr_graph_index))
                diff_fetch.append(diff_level)

            input_tensors.append(tree_graphs)
            input_tensors.append(tree_graph_edges)
            input_tensors.append(embedding_fetch)
            input_tensors.append(diff_fetch)
        
        result = sess.run(input_tensors, feed_dict)
        
        combined_loss.append(result[1])
        meta_train_loss.append(result[3])
        meta_train_acc.append(result[5])
        meta_test_acc.append(result[6])
        attention.append(result[7])
        meta_graph.append(result[8])
        meta_graph_edges.append(result[9])
        graph_embeddings.append(result[10])
        embed_diffs.append(result[11])

        if FLAGS.DEBUG == True:

            print("\n\n\n Epoch {}\n".format(itr))
            # Compare the graphs between each level
            random_idx = -1
            print("\n")
            for i, num_graph in enumerate(FLAGS.graph_list):
                if i == 0: continue
                for j in range(FLAGS.graph_list[i-1]):
                    prev_embed = np.array(graph_embeddings[random_idx][i-1][j])
                    soft_att = []
                    for k in range(num_graph):
                        curr_graph = np.array(meta_graph[random_idx][i][k])
                        curr_edges = np.array(meta_graph_edges[random_idx][i][k])
                        curr_embed = graph_embedding(curr_graph, curr_edges)
                        soft_att.append(np.exp(-np.sum(np.square(curr_embed- prev_embed)) / (2.0 * 2.0)))
                    soft_att = soft_att/np.sum(soft_att)
                    print("level {} graph {} attention with level {}: {}".format(i-1,j,i,soft_att))

            print("\n")

            for level, level_diff in enumerate(embed_diffs[-1]):
                for idx, diff in enumerate(level_diff):
                    print("Euclidean diff between embed of level {} graph {} and level {}: {}".format(level, idx, level+1, diff))

            for i in range(len(attention[-1])):
                print("Attention between level {} and {}: {}".format(i, i+1, attention[-1][i]))
            if itr >3:
                assert False

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iter {}'.format(itr)
            std = np.std(meta_test_acc, 0)
            ci95 = 1.96 * std / np.sqrt(PRINT_INTERVAL)
            print_str += ': metaTrainAcc: ' + str(np.mean(meta_train_acc)) + ', metaTestAcc: ' + str(
                np.mean(meta_test_acc)) + ', metaTrainLoss: ' + str(np.mean(meta_train_loss)) + ', combinedLoss: ' + str(
                np.mean(combined_loss)) + ', confidence: ' + str(ci95) + ', timeTaken: ' + str(
                time.time() - start_time) + ' secs, estimatedRemainingTime: ' + str(
                (time.time() - start_time)*((FLAGS.metatrain_iterations-itr)/PRINT_INTERVAL)/3600) + ' hrs'

            print(print_str)
            start_time = time.time()
            meta_train_acc, meta_test_acc, meta_train_loss, combined_loss = [], [], [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def graph_embedding(graph_nodes, graph_edges):
    graph_nodes = graph_nodes.squeeze(axis=1)
    node_size = graph_edges.shape[0]
    I = np.eye(node_size)
    adj = graph_edges + I
    D = np.diag(np.sum(adj, axis=1))
    adj = np.matmul(np.linalg.inv(D), adj)
    feature_matrix = np.matmul(adj, graph_nodes)
    embedding = np.mean(feature_matrix, axis=0)
    return  embedding


def test(model, sess, data_generator):
    num_classes = data_generator.num_classes

    metaval_accuracies = []

    graph_emb =[]
    attention =[]
    meta_graph = []
    meta_graph_edges = []
    print("Total number of test iterations ", format(FLAGS.num_test_task))

    for test_itr in range(FLAGS.num_test_task):
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}
        else:
            feed_dict = {model.meta_lr: 0.0}

        if model.classification:
            fetch = [[model.metaval_total_accuracy1] + model.metaval_total_accuracies2]
            
            embedding_fetch = []
            for level, graphs in enumerate(FLAGS.graph_list):
                embedding_level = []
                for graph_idx in range(graphs):
                        if level == 0:
                            embedding_level.append('model/level_{}_graph_{}_embedding:0'.format(level, graph_idx))
                            pass
                        else:
                            embedding_level.append('model/level_{}_graph_{}_embedding_1:0'.format(level, graph_idx))
                embedding_fetch.append(embedding_level)
            fetch.append(embedding_fetch)

            attention_fetch = []
            for level, graphs in enumerate(FLAGS.graph_list[:-1]):
                attention_level = []
                for graph_idx in range(graphs):
                    attention_level.append('model/attention_l{}n{}_to_l{}:0'.format(level, graph_idx, level+1))
                attention_fetch.append(attention_level)
            fetch.append(attention_fetch)

            tree_graphs = []
            tree_graphs_edges = []
            for level, num_graphs in enumerate(FLAGS.graph_list):
                graph_list = []
                graph_edges_list = []
                for graph_idx in range(num_graphs):
                    node_list = []
                    graph_edges_list.append('meta_graph_edges_level_{}_graph_{}:0'.format(level, graph_idx))
                    for node in range(FLAGS.num_graph_vertex):
                        node_list.append('level_{}_graph_{}_{}_node_cluster_center:0'.format(level, graph_idx, node))
                    graph_list.append(node_list)
                tree_graphs.append(graph_list)
                tree_graphs_edges.append(graph_edges_list)
            fetch.append(tree_graphs)
            fetch.append(tree_graphs_edges)

            result = sess.run(fetch, feed_dict)
        else:
            result = sess.run([model.metaval_total_loss1] + model.metaval_total_losses2, feed_dict)

        metaval_accuracies.append(result[0])
        graph_emb.append(result[1])
        attention.append(result[2])
        meta_graph.append(result[3])
        meta_graph_edges.append(result[4])

    print("\n")
    random_idx = random.randint(0,FLAGS.num_test_task)
    attention_sample = 100

    att_dict = {}
    for idx in range(100):
        for i in range(len(attention[0])):
            if i not in att_dict: att_dict[i] = {}
            for j in range(len(attention[0][i])):
                if j not in att_dict[i]: att_dict[i][j] = []
                att_dict[i][j].append(attention[idx][i][j])

    print('\n')
    print("Attention evaluated for a random task\n")    
    for i in range(len(attention[0])):
        for j in range(len(attention[0][i])): 
            print("attention between level {} and level {} graph {}: {}".format(i, i+1, j, attention[random_idx][i][j]))
    
    print("\n Attention evaluated for {} test tasks\n".format(attention_sample))    
    for i in range(len(attention[0])):
        for j in range(len(attention[0][i])): 
            print("attention between level {} and level {} graph {}: mean: {} std: {}".format(i, i+1, j, np.mean(np.array(att_dict[i][j]), axis=0), np.std(np.array(att_dict[i][j]), axis=0)))
    print("\n")
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(FLAGS.num_test_task)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def main():
    sess = tf.InteractiveSession()
    if FLAGS.train:
        test_num_updates = FLAGS.num_updates
    else:
        test_num_updates = FLAGS.num_updates_test

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource in ['2D']:
        data_generator = DataGenerator(FLAGS.update_batch_size + FLAGS.update_batch_size_eval, FLAGS.meta_batch_size)
    else:
        if FLAGS.train:
            data_generator = DataGenerator(FLAGS.update_batch_size + 15,
                                           FLAGS.meta_batch_size)
        else:
            data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                           FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    # Slice the images into input a and b
    # input a shape: batch_size, num_class*update_batch_size,im_w*im_h*3
    # label a shape: batch_size, num_class*update_batch_size, num_class
    if FLAGS.datasource in ['plainmulti', 'artmulti']:
        num_classes = data_generator.num_classes
        if FLAGS.train:
            random.seed(5)
            if FLAGS.datasource == 'plainmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_plainmulti()
            elif FLAGS.datasource == 'artmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_artmulti()
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
        else:
            random.seed(6)
            if FLAGS.datasource == 'plainmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_plainmulti(train=False)
            elif FLAGS.datasource == 'artmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_artmulti(train=False)
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        input_tensors = None
        metaval_input_tensors=None
    # Initialize the MAML model along with LSTM AE, prototype graph and knowledge graph 
    model = MAML(sess, dim_input, dim_output, test_num_updates=test_num_updates)

    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=60)

    if FLAGS.train == False:
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
        FLAGS.update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.emb_loss_weight' + str(
        FLAGS.emb_loss_weight) + '.hidden_dim' + str(FLAGS.hidden_dim)

    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
        if model_file:
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    resume_itr = FLAGS.test_epoch


    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, sess, data_generator)

if __name__ == "__main__":
    main()
