'''This script demonstrates how to build a variational autoencoder with Keras.
    #Reference
    - Auto-Encoding Variational Bayes
    https://arxiv.org/abs/1312.6114
    '''
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, Flatten, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Embedding, GlobalAveragePooling1D, Reshape
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix

import argparse
import random
import io,os

from operator import itemgetter
from itertools import groupby
import statistics as stats

import sqlite3
import csv
import collections

batch_size = 1
original_dim = 784  # data size of each sample
latent_dim = 3
intermediate_dim = 5
epochs = 50
epsilon_std = 1.0

cur_relations = []

choices = ( # possible block pairs
           (1,2),(1,3),(1,4),(1,5),(1,6),
           (2,1),(2,3),(2,4),(2,5),(2,6),
           (3,1),(3,2),(3,4),(3,5),(3,6),
           (4,1),(4,2),(4,3),(4,5),(4,6),
           (5,1),(5,2),(5,3),(5,4),(5,6),
           (6,1),(6,2),(6,3),(6,4),(6,5),
           (0,0)
           )

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reversed_dictionary[valid_examples[i]]
            top_k = 6  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s (%s):' % (valid_word, valid_examples[i])
            for k in range(top_k):
                close_word = reversed_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((len(vocab),))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(len(vocab)):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

sim_cb = SimilarityCallback()

def main():
    parser = argparse.ArgumentParser(description='Classifier for EMRE')
    parser.add_argument('-d', '--database', metavar='DATABASE', help='database file')
    parser.add_argument('-b', '--batch', metavar='BATCH', help='batch data csv')
    parser.add_argument('-f', '--formal', action='store_true', help='use formal features')
    parser.add_argument('-w', '--word_embeddings', action='store_true', help='use word embeddings')
    parser.add_argument('-F', '--formal_only', action='store_true', help='use formal features only')
    parser.add_argument('-r', '--retrain_word', action='store_true', help='retrain word embeddings')
    parser.add_argument('-R', '--retrain_model', action='store_true', help='retrain model')
    parser.add_argument('-s', '--smooth', action='store_true', help='rebalance sample')
    parser.add_argument('-L', '--ling_only', action='store_true', help='linguistic REs only')
    parser.add_argument('-E', '--ens_only', action='store_true', help='ensemble REs only')
    parser.add_argument('-k', '--k', metavar='K', help='cross-validation K')
    parser.add_argument('-K', '--offset', metavar='OFFSET', help='cross-validation offset')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    args = parser.parse_args()
    
    db = args.database
    batch = args.batch
    formal = args.formal
    formal_only = args.formal_only
    embeddings = args.word_embeddings
    retrain_word = args.retrain_word
    retrain_model = args.retrain_model
    rebalance = args.smooth
    ling_only = args.ling_only
    ens_only = args.ens_only
    k = int(args.k)
    offset = int(args.offset)
    verbose = args.verbose
    
    if ling_only and ens_only:
        print("Cannot evaluate ONLY linguistic REs and ONLY ensemble REs! (pick one)")
        return
    
    if formal_only:
        formal = True

    global embedding_dim
    embedding_dim = 200

    hit_results = []

    if os.path.isfile(batch):
        file = open(batch)
        i = 0
        for entry in csv.reader(file):
            if entry[0] != "HITId":
                vidA = entry[27].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankA = int(entry[37])
                vidB = entry[29].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankB = int(entry[38])
                vidC = entry[31].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankC = int(entry[39])
                vidD = entry[33].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankD = int(entry[40])
                vidE = entry[35].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankE = int(entry[41])
                
                if ling_only:
                    hit_results.append([vidB,vidC,
                                        rankA,rankB,rankC,rankD,rankE])
                elif ens_only:
                    hit_results.append([vidD,vidE,
                                    rankA,rankB,rankC,rankD,rankE])
                else:
                    hit_results.append([vidA,vidB,vidC,vidD,vidE,
                                    rankA,rankB,rankC,rankD,rankE])
                                    
                i += 1
    else:
        print("%s is not a file" % batch)
        exit()
    
    print(hit_results)

    global vocab
    all_words = []
    vocab = []

    vid_rankings = {}
    vid_features = {}
    vid_formal_features = {}

    if os.path.isfile(db):
        connection = sqlite3.connect(db)
        cursor = connection.cursor()
        
        # get all results in db
        cmd = "SELECT * FROM EMREVideoDBEntry"
        print(cmd)
        cursor.execute(cmd)
        all_results = [r for r in cursor.fetchall()]
        print(all_results)
        
        # gather all individual words used in RE DescriptionStrs (index 5)
        for result in all_results:
            words = result[5].lower().strip('.').replace(',','').split()
            for word in words:
                all_words.append(str(word))
                if word not in vocab:
                    vocab.append(str(word))

        for result in hit_results:
            # result = [vidA,vidB,vidC,vidD,vidE,rankA,rankB,rankC,rankD,rankE]
            #print(result)
            j = 5
            if ling_only:
                j = 2
                rankings = result[:2] + [(-r+6 if result[2] > result[6] else r) for r in result[2:]][1:3]
            elif ens_only:
                j = 2
                rankings = result[:2] + [(-r+6 if result[2] > result[6] else r) for r in result[2:]][3:]
            else:
                rankings = result[:5] + [(-r+6 if result[5] > result[9] else r) for r in result[5:]]
            
            for i in range(0,j):
                filename = rankings[i]
                cmd = "SELECT * FROM EMREVideoDBEntry" + " WHERE FilePath = '" + filename + "'"
                cursor.execute(cmd)
                vid_result = cursor.fetchone()
                vid_features[filename] = vid_result
                if formal:
                    block_to_attr = {
                        "red_block1" : "red",
                        "purple_block3" : "purple",
                        "block4" : "green",
                        "green_block5" : "green",
                        "block6" : "red",
                        "block7" : "purple"
                    }
                    attr_words = ["red","green","purple"]
                    spatial_words = ["left","right","front","behind"]
                    formal_features = []
                    desc_words = vid_features[filename][5].lower().strip('.').replace(',','').split()
                    formal_features.append(len([s for s in spatial_words if s in desc_words]))
                    formal_features.append(len([a for a in attr_words if a in desc_words]))

#                    if 'other' in desc_words:
#                        formal_features.append(1)
#                    else:
#                        formal_features.append(0)
#
#                    if 'this' in desc_words:
#                        formal_features.append(1)
#                    else:
#                        formal_features.append(0)

                    if str(vid_features[filename][4]) == "Gestural" or str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)
                    
                    if str(vid_features[filename][4]) == "Linguistic" or str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)

                    if 'this' in desc_words and str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)
                    
                    if ('this' in desc_words or 'that' in desc_words) and 'other' in desc_words and  str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)
                    
                    if 'other' in desc_words and str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)

                    if (len([a for a in attr_words if a in desc_words and a != block_to_attr[str(vid_features[filename][2])]]) > 0) and ('this' in desc_words or 'that' in desc_words):
                        formal_features.append(1)
                    else:
                        formal_features.append(0)

                    if (len([a for a in attr_words if a in desc_words and a != block_to_attr[str(vid_features[filename][2])]]) > 0) and ('this' in desc_words or 'that' in desc_words) and str(vid_features[filename][4]) == "Ensemble":
                        formal_features.append(1)
                    else:
                        formal_features.append(0)

                    if len(vid_features[filename][10]) == 0:
                        formal_features.append(0)
                    else:
                        formal_features.append(vid_features[filename][10].count('\n')+1)
                    
                    vid_formal_features[filename] = formal_features
                if (filename not in vid_rankings):
                    if verbose:
                        print("Adding vid_rankings[%s]" % filename)
                    vid_rankings[filename] = np.zeros((5,))
                if verbose:
                    print(filename, i, rankings[i+j]-1)
                print(rankings)
                vid_rankings[filename][rankings[i+j]-1] += 1
                if verbose:
                    print("vid_rankings[%s][%s] = %s" % (filename,result[i+j]-1,vid_rankings[filename][rankings[i+j]-1]))
                    print(filename, vid_rankings[filename])

        for filename in sorted(vid_rankings.keys()):
            #vid_rankings[filename] = [r-stats.median(vid_rankings[filename]) for r in vid_rankings[filename]]
            print(filename, vid_rankings[filename])

        connection.close()
    else:
        print("%s is not a file" % db)
    
#return

    print(all_words)
    print(vocab)

    global data
    global count
    global dictionary
    global reversed_dictionary
    data, count, dictionary, reversed_dictionary = build_dataset(all_words)
    print(data)
    print(count)
    print(dictionary)
    print(reversed_dictionary)

    obj_to_int = {
        "red_block1" : 0,
        "purple_block3" : 1,
        "block4" : 2,
        "green_block5" : 3,
        "block6" : 4,
        "block7" : 5
    }
    
    mod_to_int = {
        "Gestural" : 0,
        "Linguistic" : 1,
        "Ensemble" : 2
    }
        
    ddtype_to_int = {
        "None" : 0,
        "Absolute" : 1,
        "Relative" : 2
    }

    if embeddings:
        if retrain_word:
            dumb_embeddings = dumb_word_embedding_shit_idegaf(vocab)
        else:
            dumb_embeddings = keras.models.load_model("dumb_embeddings")

        weights = dumb_embeddings.layers[2].get_weights()[0]
        words_embeddings = {w : weights[idx] for w, idx in dictionary.items()}
        
        for word in vocab:
            print(word, words_embeddings[word].shape, words_embeddings[word])

        sent_embeddings = {}
        rel_embeddings = {}

        for result in all_results:
            desc_str = result[5].lower().strip('.').replace(',','')
            words = desc_str.split()
            desc_str_vector = np.zeros((len(words_embeddings[vocab[0]]),),dtype=np.float64)
            for word in words:
                desc_str_vector += words_embeddings[word]
            if verbose:
                print(desc_str,desc_str_vector)
            sent_embeddings[desc_str] = desc_str_vector
            relation_desc = result[10].lower().replace('\n',' ')
            words = relation_desc.split()
            relation_desc_vector = np.zeros((len(words_embeddings[vocab[0]]),),dtype=np.float64)
            for word in words:
                relation_desc_vector += words_embeddings[word]
            if verbose:
                print(relation_desc,relation_desc_vector)
            rel_embeddings[relation_desc] = relation_desc_vector

    if retrain_model:
        test_vids = vid_features.keys()[offset::k]
        train_vids = vid_features.keys()
        del train_vids[offset::k]

        if formal_only:
            train_features_size = 0
        else:
            if ling_only:
                train_features_size = 1
            elif ens_only:
                train_features_size = 4
            else:
                train_features_size = 5
        
        if formal:
            train_features_size += len(vid_formal_features[vid_formal_features.keys()[0]])
        if embeddings:
            train_features_size += (2*words_embeddings[words_embeddings.keys()[0]].shape[0])

        mlp_x_train = np.zeros((len(train_vids),train_features_size),dtype=np.float64)
        mlp_y_train = np.zeros((len(train_vids),5),dtype=np.int64)

        print(vid_features)
        print(vid_rankings)
        
        print(len(vid_features))
        print(len(vid_rankings))
        
        if formal:
            print(vid_formal_features)
            print(len(vid_formal_features))
        
        print(mlp_x_train.shape)
        print(mlp_y_train.shape)

        # FocusObj (int), RefModality (int), DescriptionStr (vector),
        # ObjDistToAgent (float), DistanceDistinction (int), DistDistinctionType (int),
        # [RelationalDescriptors (vector?)]

        row = 0
        last_column = 0
        for vidname in train_vids:
            if formal_only:
                last_column = 0
            else:
                if ling_only:
                    mlp_x_train[row][0] = obj_to_int[str(vid_features[vidname][2])]
                    if embeddings:
                        mlp_x_train[row][1:1+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                        mlp_x_train[row][1+embedding_dim:1+(2*embedding_dim)] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                    last_column = 1+(2*(int(embeddings)*embedding_dim))
                elif ens_only:
                    mlp_x_train[row][0] = obj_to_int[str(vid_features[vidname][2])]
                    if embeddings:
                        mlp_x_train[row][1:1+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                    mlp_x_train[row][1+(int(embeddings)*embedding_dim)] = float(vid_features[vidname][7])
                    mlp_x_train[row][1+(int(embeddings)*embedding_dim)+1] = int(vid_features[vidname][8])
                    mlp_x_train[row][1+(int(embeddings)*embedding_dim)+2] = ddtype_to_int[str(vid_features[vidname][9])]
                    if embeddings:
                        mlp_x_train[row][1+(int(embeddings)*embedding_dim)+3:1+\
                                         (2*(int(embeddings)*embedding_dim))+3] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                    last_column = 1+(2*(int(embeddings)*embedding_dim))+3
                else:
                    mlp_x_train[row][0] = obj_to_int[str(vid_features[vidname][2])]
                    mlp_x_train[row][1] = mod_to_int[str(vid_features[vidname][4])]
                    if embeddings:
                        mlp_x_train[row][2:2+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                    mlp_x_train[row][2+(int(embeddings)*embedding_dim)] = float(vid_features[vidname][7])
                    mlp_x_train[row][2+(int(embeddings)*embedding_dim)+1] = int(vid_features[vidname][8])
                    mlp_x_train[row][2+(int(embeddings)*embedding_dim)+2] = ddtype_to_int[str(vid_features[vidname][9])]
                    if embeddings:
                        mlp_x_train[row][2+(int(embeddings)*embedding_dim)+3:2+\
                                         (2*(int(embeddings)*embedding_dim))+3] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                    last_column = 2+(2*(int(embeddings)*embedding_dim))+3
            if formal:
                mlp_x_train[row][last_column:] = np.asarray(vid_formal_features[vidname])
            row += 1

        row = 0
        for vidname in train_vids:
            if rebalance:
                mean = np.sum(np.asarray([vid_rankings[vidname][i]*i for i in range(len(vid_rankings[vidname]))]))/np.sum(vid_rankings[vidname])
                if np.argmax(vid_rankings[vidname]) == 0:
                    if mean < 1.875:
                        mlp_y_train[row][0] = 1
                    else:
                        mlp_y_train[row][1] = 1
                elif np.argmax(vid_rankings[vidname]) == 1:
                    if mean < 2.085:
                        mlp_y_train[row][1] = 1
                    else:
                        mlp_y_train[row][2] = 1
                elif np.argmax(vid_rankings[vidname]) == 2:
                    if mean > 2.28:
                        mlp_y_train[row][3] = 1
                    else:
                        mlp_y_train[row][2] = 1
                elif np.argmax(vid_rankings[vidname]) == 3:
                    if mean > 2.405:
                        mlp_y_train[row][4] = 1
                    else:
                        mlp_y_train[row][3] = 1
                elif np.argmax(vid_rankings[vidname]) == 4:
                    mlp_y_train[row][4] = 1
            else:
                mlp_y_train[row][np.argmax(vid_rankings[vidname])] = 1
            row += 1

        print(mlp_x_train.shape, mlp_x_train)
        print(mlp_y_train.shape, mlp_y_train)

        if formal_only:
            input_size = 0
        else:
            if ling_only:
                input_size = 1
            elif ens_only:
                input_size = 4
            else:
                input_size = 5

        if formal:
            input_size += len(vid_formal_features[vid_formal_features.keys()[0]])
        if embeddings:
            input_size += (2*words_embeddings[words_embeddings.keys()[0]].shape[0])

        mlp = Sequential()
        mlp.add(Dense(32, activation='tanh', input_dim=input_size))
        #mlp.add(Dropout(0.2))
        mlp.add(Dense(128, activation='elu'))
##    #    mlp.add(Dropout(0.2))
        mlp.add(Dense(64, activation='tanh'))
##    #    mlp.add(Dropout(0.2))
#        mlp.add(Dense(64, activation='tanh'))
#        mlp.add(Dense(128, activation='tanh'))
#        mlp.add(Dense(64, activation='tanh'))
        mlp.add(Dense(5, activation='softmax'))

        sgd = SGD(lr=0.1, decay=0, momentum=0.0, nesterov=False)
        mlp.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
            
        mlp.fit(mlp_x_train, mlp_y_train,
                epochs=500,
                batch_size=100)
                
        mlp.save("emre_mlp")
    else:
        mlp = keras.models.load_model("emre_mlp")

    if formal_only:
        test_features_size = 0
    else:
        if ling_only:
            test_features_size = 1
        elif ens_only:
            test_features_size = 4
        else:
            test_features_size = 5

    if formal:
        test_features_size += len(vid_formal_features[vid_formal_features.keys()[0]])
    if embeddings:
        test_features_size += (2*words_embeddings[words_embeddings.keys()[0]].shape[0])

    mlp_x_test = np.zeros((len(test_vids),test_features_size),dtype=np.float64)
    #mlp_x_test = np.zeros((1350,5),dtype=np.float64)

    row = 0
    last_column = 0
    for vidname in test_vids:
        if formal_only:
            last_column = 0
        else:
            if ling_only:
                mlp_x_test[row][0] = obj_to_int[str(vid_features[vidname][2])]
                if embeddings:
                    mlp_x_test[row][1:1+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                    mlp_x_test[row][1+embedding_dim:1+(2*embedding_dim)] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                last_column = 1+(2*(int(embeddings)*embedding_dim))
            elif ens_only:
                mlp_x_test[row][0] = obj_to_int[str(vid_features[vidname][2])]
                if embeddings:
                    mlp_x_test[row][1:1+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                mlp_x_test[row][1+(int(embeddings)*embedding_dim)] = float(vid_features[vidname][7])
                mlp_x_test[row][1+(int(embeddings)*embedding_dim)+1] = int(vid_features[vidname][8])
                mlp_x_test[row][1+(int(embeddings)*embedding_dim)+2] = ddtype_to_int[str(vid_features[vidname][9])]
                if embeddings:
                    mlp_x_test[row][1+(int(embeddings)*embedding_dim)+3:1+\
                                    (2*(int(embeddings)*embedding_dim))+3] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                last_column = 1+(2*(int(embeddings)*embedding_dim))+3
            else:
                mlp_x_test[row][0] = obj_to_int[str(vid_features[vidname][2])]
                mlp_x_test[row][1] = mod_to_int[str(vid_features[vidname][4])]
                if embeddings:
                    mlp_x_test[row][2:2+embedding_dim] = sent_embeddings[str(vid_features[vidname][5].lower().strip('.').replace(',',''))]
                mlp_x_test[row][2+(int(embeddings)*embedding_dim)] = float(vid_features[vidname][7])
                mlp_x_test[row][2+(int(embeddings)*embedding_dim)+1] = int(vid_features[vidname][8])
                mlp_x_test[row][2+(int(embeddings)*embedding_dim)+2] = ddtype_to_int[str(vid_features[vidname][9])]
                if embeddings:
                    mlp_x_test[row][2+(int(embeddings)*embedding_dim)+3:2+\
                                (2*(int(embeddings)*embedding_dim))+3] = rel_embeddings[str(vid_features[vidname][10].lower().replace('\n',' '))]
                last_column = 2+(2*(int(embeddings)*embedding_dim))+3
        if formal:
            mlp_x_test[row][last_column:] = np.asarray(vid_formal_features[vidname])
        row += 1

    labels = []
    for vidname in test_vids:
        if rebalance:
            mean = np.sum(np.asarray([vid_rankings[vidname][i]*i for i in range(len(vid_rankings[vidname]))]))/np.sum(vid_rankings[vidname])
            if np.argmax(vid_rankings[vidname]) == 0:
                if mean < 1.6:
                    labels.append(0)
                else:
                    labels.append(1)
            elif np.argmax(vid_rankings[vidname]) == 1:
                if mean < 2.0:
                    labels.append(1)
                else:
                    labels.append(2)
            elif np.argmax(vid_rankings[vidname]) == 2:
                if mean < 2.1:
                    labels.append(2)
                else:
                    labels.append(3)
            elif np.argmax(vid_rankings[vidname]) == 3:
                if mean < 2.8:
                    labels.append(3)
                else:
                    labels.append(4)
            elif np.argmax(vid_rankings[vidname]) == 4:
                labels.append(4)
        else:
            labels.append(np.argmax(vid_rankings[vidname]))

    print(mlp_x_test,mlp_x_test.shape)

    results = np.argmax(mlp.predict(mlp_x_test),axis=1)
        
    print("Predicted: %s, Ground truth: %s" % (results,labels))

    print(mlp.summary())

    interval_results = []
    for i in range(len(labels)):
        if labels[i] == 0:
            if results[i] == 0 or results[i] == 1:
                interval_results.append(2)    # in quintile
            elif results[i] == 2:
                interval_results.append(3)    # above quintile +1
            elif results[i] == 3 or results[i] == 4:
                interval_results.append(4)    # above quintile +2
        elif labels[i] == 1:
            if results[i] == 0 or results[i] == 1 or results[i] == 2:
                interval_results.append(2)    # in quintile
            elif results[i] == 3:
                interval_results.append(3)    # above quintile +1
            elif results[i] == 4:
                interval_results.append(4)    # above quintile +2
        elif labels[i] == 2:
            if results[i] == 0:
                interval_results.append(1)    # below quintile -1
            elif results[i] == 1 or results[i] == 2 or results[i] == 3:
                interval_results.append(2)    # in quintile
            elif results[i] == 4:
                interval_results.append(3)    # above quintile +1
        elif labels[i] == 3:
            if results[i] == 0:
                interval_results.append(0)    # below quintile -2
            elif results[i] == 1:
                interval_results.append(1)    # below quintile -1
            elif results[i] == 2 or results[i] == 3 or results[i] == 4:
                interval_results.append(2)    # in quintile
        elif labels[i] == 4:
            if results[i] == 0 or results[i] == 1:
                interval_results.append(0)    # below quintile -2
            elif results[i] == 2:
                interval_results.append(1)    # below quintile -1
            elif results[i] == 3 or results[i] == 4:
                interval_results.append(2)    # in quintile

    print(confusion_matrix(labels, results))
    print(classification_report(labels, results))

    print(confusion_matrix(interval_results, [2 for i in range(len(interval_results))]))
    print(classification_report(interval_results, [2 for i in range(len(interval_results))]))

    return
    
    if os.path.exists("%s/Block%sPerson%s.csv" % (data,block,person)):
        mlp_x_train = np.zeros((1350,7),dtype=np.float64)
        mlp_y_train = np.zeros((1350,5),dtype=np.int64)
        for filepath in os.listdir(data):
            if not filepath.startswith('.'):
                cell = list(map(int,filepath.split('.')[0].replace("Block","").replace("Person",",").split(',')))
                row = (cell[0]-1)*20+(cell[1]-1)
                print(filepath,cell,row)
                content = open("%s/%s" % (data,filepath)).readlines()[0].strip().split(',')
                print(content)
                features = list(map(float,content[1:-1]))
                label = list(map(int,content[-1]))
                print(features,label)
                mlp_x_train[row] = np.asarray(features)
                mlp_y_train[row][label] = 1
        print(mlp_x_train)
        print(mlp_y_train)
        
        mlp = Sequential()
        # Dense(64) is a fully-connected layer with 12 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 7-dimensional vectors.
        # FocusObj (int), RefModality (int), DescriptionStr (vector?),
        # ObjDistToAgent (float), DistanceDistinction (int), DistDistinctionType (int),
        # RelationalDescriptors (vector?)
        mlp.add(Dense(128, activation='relu', input_dim=12))
#        mlp.add(Dropout(0.5))
        mlp.add(Dense(128, activation='relu'))
#        mlp.add(Dropout(0.5))
        mlp.add(Dense(128, activation='relu'))
#        mlp.add(Dropout(0.5))
        mlp.add(Dense(128, activation='relu'))
#        mlp.add(Dropout(0.5))
        mlp.add(Dense(3, activation='sigmoid'))
        
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        mlp.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        mlp.fit(mlp_x_train, mlp_y_train,
                epochs=20,
                batch_size=5)

        filepath = "Block%sPerson%s.csv" % (block,person)
        print(filepath)

        cell = list(map(int,filepath.split('.')[0].replace("Block","").replace("Person",",").split(',')))
        row = (cell[0]-1)*20+(cell[1]-1)
        print(filepath,cell,row)
        content = open("%s/%s" % (data,filepath)).readlines()[0].strip().split(',')
        print(content)
        features = list(map(float,content[1:-1]))
        label = list(map(int,content[-1]))
        print(features,label)
        
        mlp_x_test = np.asarray((features,),dtype=np.float64)

        print(mlp_x_test,mlp_x_test.shape)

        results = mlp.predict(mlp_x_test)

        print("Predicted: %s, Correct: %s" % (np.argmax(results,axis=1),label))

    return

def dumb_word_embedding_shit_idegaf(vocab):
    window_size = 2
    vector_dim = embedding_dim
    epochs = 50000
    
    global valid_size
    global valid_examples
    valid_size = 6     # Random set of words to evaluate similarity on.
    valid_window = 10  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    
    sampling_table = sequence.make_sampling_table(len(vocab))
    print(sampling_table)
    couples, labels = sequence.skipgrams(data, len(vocab), window_size=window_size, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    
    print(couples[:10], labels[:10])

    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))
    
    embedding = Embedding(len(vocab), vector_dim, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)
    
    # setup a cosine similarity operation which will be output in a secondary model
    similarity = keras.layers.dot([target, context], 1)
    
    # now perform the dot product operation to get a similarity measure
    dot_product = keras.layers.dot([target, context], 1, normalize=True)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    
    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    # create a secondary validation model to run our similarity checks during training
    global validation_model
    validation_model = Model(input=[input_target, input_context], output=similarity)
    
    arr_1 = np.zeros((1,))
    arr_2 = np.zeros((1,))
    arr_3 = np.zeros((1,))
    for cnt in range(epochs):
        idx = np.random.randint(0, len(labels)-1)
        arr_1[0,] = word_target[idx]
        arr_2[0,] = word_context[idx]
        arr_3[0,] = labels[idx]
        loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if cnt % 100 == 0:
            print("Iteration {}, loss={}".format(cnt, loss))
        if cnt % 10000 == 0:
            sim_cb.run_sim()

    sim_cb.run_sim()

    model.save("dumb_embeddings")

    return model

def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = []
    count.extend(collections.Counter(words))
    dictionary = dict()
    for word in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

if __name__ == "__main__":
    main()
