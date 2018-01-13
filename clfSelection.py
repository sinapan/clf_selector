#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from sklearn import preprocessing
from sklearn.utils import shuffle

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN #RandomOverSampler #SMOTE #ADASYN
#from imblearn.under_sampling import TomekLinks

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

##############################################################################
# Data
##############################################################################
# Import data
CSV_IN   = pd.read_csv('featureSheet.csv',index_col=0)
# Input data
X_mat_original = CSV_IN.loc[:,('num_of_sections','num_of_rows','num_of_seats','lastSRSCount','num_of_fbPosts','num_of_twPosts','num_of_snapshots')]
# Input Labels
Y_mat_original = CSV_IN.loc[:,('labels')]
# Convert to numpy matrix
CSV_IN['labels'] = CSV_IN['labels'].astype(int)
CSV_IN_mat= CSV_IN.as_matrix()
X_mat_original   = X_mat_original.as_matrix()
Y_mat_original   = Y_mat_original.as_matrix()

##############################################################################
# Balancing
##############################################################################
# Apply the random over-sampling
ada = ADASYN() #RandomOverSampler() #SMOTE() #ADASYN()
X_mat, Y_mat = ada.fit_sample(X_mat_original, Y_mat_original)
#X_mat, Y_mat = balanceMATs(X_mat_original, Y_mat_original)

# Randomizing labels for sanity check
Y_mat_rnd = shuffle(Y_mat)
Y_mat_rnd = shuffle(Y_mat_rnd)
Y_mat_rnd = shuffle(Y_mat_rnd)

##############################################################################
# Model
##############################################################################
# Model Characteristics
num_samples      = X_mat.shape[0]
num_classes      = np.bincount(Y_mat).shape[0]
num_true_samples = np.bincount(Y_mat)[1]
num_PCA          = 7  #0:5features; 6:all;
norm_num         = 2
k_fold           = 10

##############################################################################
# For debug
##############################################################################
#pcaCounter = 0;

##############################################################################
# Normalization
##############################################################################
# 0:no scaling
# 1:unit norm samples; 2:zero mean and unit variance
# 3:zero to one scale; 4:negative one to positive one scale
# 5:2&3;               6:2&4
def norm_data(norm_num, X_train_or, X_test_or):
    if norm_num == 0:
        X_tr_norm = X_train_or
        X_ts_norm = X_test_or
    elif norm_num == 1:
        scaler    = preprocessing.Normalizer().fit(X_train_or)
        X_tr_norm = scaler.transform(X_train_or)
        X_ts_norm = scaler.transform(X_test_or)
    elif norm_num == 2:
        scaler    = preprocessing.StandardScaler().fit(X_train_or)
        X_tr_norm = scaler.transform(X_train_or)
        X_ts_norm = scaler.transform(X_test_or)            
    elif norm_num == 3:
        scaler    = preprocessing.MinMaxScaler().fit(X_train_or)
        X_tr_norm = scaler.transform(X_train_or)
        X_ts_norm = scaler.transform(X_test_or)
    elif norm_num == 4:
        scaler    = preprocessing.MaxAbsScaler().fit(X_train_or)
        X_tr_norm = scaler.transform(X_train_or)
        X_ts_norm = scaler.transform(X_test_or)
    elif norm_num == 5:
        scaler1    = preprocessing.StandardScaler().fit(X_train_or)
        X_tr_norm1 = scaler1.transform(X_train_or)
        X_ts_norm1 = scaler1.transform(X_test_or)
        scaler2    = preprocessing.MinMaxScaler().fit(X_tr_norm1)
        X_tr_norm  = scaler2.transform(X_tr_norm1)
        X_ts_norm  = scaler2.transform(X_ts_norm1)
    elif norm_num == 6:
        scaler1    = preprocessing.StandardScaler().fit(X_train_or)
        X_tr_norm1 = scaler1.transform(X_train_or)
        X_ts_norm1 = scaler1.transform(X_test_or)
        scaler2    = preprocessing.MaxAbsScaler().fit(X_tr_norm1)
        X_tr_norm  = scaler2.transform(X_tr_norm1)
        X_ts_norm  = scaler2.transform(X_ts_norm1)
    return X_tr_norm, X_ts_norm

##############################################################################
# Graph
##############################################################################
def rungraph(solver_method, NN_n, X_tr, X_ts, y_tr, y_ts):
    # Classifier
    if   solver_method == 'NN_lbfgs_1':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=5000, shuffle=True, early_stopping=True)
    elif solver_method == 'NN_lbfgs_2':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='tanh',     solver='lbfgs', learning_rate='constant', max_iter=5000, shuffle=True, early_stopping=True)
    elif solver_method == 'NN_lbfgs_3':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='relu',     solver='lbfgs', learning_rate='constant', max_iter=5000, shuffle=True, early_stopping=True)   
    elif solver_method == 'NN_sgd_1':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='logistic', solver='sgd',   learning_rate='adaptive', max_iter=10000, shuffle=True, early_stopping=True)
    elif solver_method == 'NN_sgd_2':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='tanh',     solver='sgd',   learning_rate='adaptive', max_iter=10000, shuffle=True, early_stopping=True)
    elif solver_method == 'NN_sgd_3':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='relu',     solver='sgd',   learning_rate='adaptive', max_iter=10000, shuffle=True, early_stopping=True)
    elif solver_method == 'NN_adam_1':
#        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='logistic', solver='adam',  learning_rate='constant', max_iter=5000, shuffle=True, early_stopping=True, learning_rate_init=0.01)
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='relu',     solver='adam',  learning_rate='constant', max_iter=100000, shuffle=True, early_stopping=False, learning_rate_init=0.01)

    elif solver_method == 'NN_adam_2':
#        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='tanh',     solver='adam',  learning_rate='constant', max_iter=5000, shuffle=True, early_stopping=True, learning_rate_init=0.01)
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='relu',     solver='adam',  learning_rate='constant', max_iter=100000, shuffle=True, early_stopping=False, learning_rate_init=0.01)

    elif solver_method == 'NN_adam_3':
        clf = MLPClassifier(hidden_layer_sizes=NN_n, activation='relu',     solver='adam',  learning_rate='constant', max_iter=100000, shuffle=True, early_stopping=False, learning_rate_init=0.01)

    elif solver_method == 'SVM_linear':
        clf = SVC(kernel='linear')
    elif solver_method == 'SVM_poly_1':
        clf = SVC(kernel='poly', degree=2)
    elif solver_method == 'SVM_poly_2':
        clf = SVC(kernel='poly', degree=3, class_weight='balanced')
    elif solver_method == 'SVM_poly_5':
        clf = SVC(kernel='poly', degree=5, class_weight='balanced')
    elif solver_method == 'SVM_rbf':
        clf = SVC(kernel='rbf')
    elif solver_method == 'SVM_sigmoid':
        clf = SVC(kernel='sigmoid', class_weight='balanced')
        
    # Fit model
    clf.fit(X_tr, y_tr)
    
    # Evaluate accuracy.
    y_pred    = clf.predict(X_ts)
    acc_score = accuracy_score(y_ts, y_pred, normalize=True)
    conf_mat  = confusion_matrix(y_ts, y_pred)
    print("still going..") 
#    if pcaCounter == 6:
#        return acc_score
    return acc_score, conf_mat

##############################################################################
# Print
##############################################################################
def printMAC(name, ac_scores):
    print(name)
    mean_ac_scores = np.mean(ac_scores, axis=0)
    sarr = [str(a) for a in np.round(mean_ac_scores*100,2)]
    print(", " . join(sarr))
    return

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    data_type_np = np.float
    xsize        = k_fold
    ysize        = num_PCA
    # init
    ac_NN3_lbfgs_1  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_lbfgs_1  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_lbfgs_1 = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_lbfgs_1 = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN3_lbfgs_2  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_lbfgs_2  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_lbfgs_2 = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_lbfgs_2 = np.zeros((xsize,ysize),dtype=data_type_np)    
    ac_NN3_lbfgs_3  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_lbfgs_3  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_lbfgs_3 = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_lbfgs_3 = np.zeros((xsize,ysize),dtype=data_type_np)
    
    ac_NN3_sgd_1    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_sgd_1    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_sgd_1   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_sgd_1   = np.zeros((xsize,ysize),dtype=data_type_np)    
    ac_NN3_sgd_2    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_sgd_2    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_sgd_2   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_sgd_2   = np.zeros((xsize,ysize),dtype=data_type_np)    
    ac_NN3_sgd_3    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_sgd_3    = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_sgd_3   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_sgd_3   = np.zeros((xsize,ysize),dtype=data_type_np)
    
    ac_NN3_adam_1   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_adam_1   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_adam_1  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_adam_1  = np.zeros((xsize,ysize),dtype=data_type_np)    
    ac_NN3_adam_2   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_adam_2   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_adam_2  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_adam_2  = np.zeros((xsize,ysize),dtype=data_type_np)    
    ac_NN3_adam_3   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN5_adam_3   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN10_adam_3  = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_NN20_adam_3  = np.zeros((xsize,ysize),dtype=data_type_np)
        
    ac_SVM_linear   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_SVM_poly_1   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_SVM_poly_2   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_SVM_poly_5   = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_SVM_rbf      = np.zeros((xsize,ysize),dtype=data_type_np)
    ac_SVM_sigmoid  = np.zeros((xsize,ysize),dtype=data_type_np)
    
    # loop over StratifiedKFold
    skf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=True)
    for i, indicies in enumerate(skf.split(X_mat, Y_mat)):
        train_index, test_index = indicies
        X_train_or, X_test_or = X_mat[train_index], X_mat[test_index]
        y_train, y_test       = Y_mat[train_index], Y_mat[test_index]
        # Normalize data
        X_train, X_test = norm_data(norm_num, X_train_or, X_test_or)
        
        # loop over different PCA
        for j in range(0, num_PCA):
#        if num_PCA == 7:
#            j = 6
            # Select Features
            # PCA
#            pcaCounter = j
            if j == 6:
                X_train_pca = X_train
                X_test_pca  = X_test
            else:
                n_PCA =j+1
                pca = PCA(n_components=n_PCA, whiten=False)
                pca.fit(X_train)
                X_train_pca = pca.transform(X_train)
                X_test_pca  = pca.transform(X_test)
            
            # loop over different hidden neurons
#            solver_method = 'NN_lbfgs_1'
#            ac_NN3_lbfgs_1[i,j]  = rungraph(solver_method, 3,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_lbfgs_1[i,j]  = rungraph(solver_method, 5,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_lbfgs_1[i,j] = rungraph(solver_method, 10, X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_lbfgs_1[i,j] = rungraph(solver_method, 20, X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'NN_lbfgs_2'
#            ac_NN3_lbfgs_2[i,j]  = rungraph(solver_method, 3,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_lbfgs_2[i,j]  = rungraph(solver_method, 5,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_lbfgs_2[i,j] = rungraph(solver_method, 10, X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_lbfgs_2[i,j] = rungraph(solver_method, 20, X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'NN_lbfgs_3'
#            ac_NN3_lbfgs_3[i,j]  = rungraph(solver_method, 3,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_lbfgs_3[i,j]  = rungraph(solver_method, 5,  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_lbfgs_3[i,j] = rungraph(solver_method, 10, X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_lbfgs_3[i,j] = rungraph(solver_method, 20, X_train_pca, X_test_pca, y_train, y_test)
            
#            solver_method = 'NN_sgd_1'
#            ac_NN3_sgd_1[i,j], temp    = rungraph(solver_method, (20,10,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_sgd_1[i,j], temp    = rungraph(solver_method, (20,30,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_sgd_1[i,j], temp   = rungraph(solver_method, (10,5,3), X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_sgd_1[i,j], temp   = rungraph(solver_method, (30,20,10), X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'NN_sgd_2'
#            ac_NN3_sgd_2[i,j], temp    = rungraph(solver_method, (20,10,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_sgd_2[i,j], temp    = rungraph(solver_method, (20,30,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_sgd_2[i,j], temp   = rungraph(solver_method, (10,5,3), X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_sgd_2[i,j], temp   = rungraph(solver_method, (30,20,10), X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'NN_sgd_3'
#            ac_NN3_sgd_3[i,j], temp    = rungraph(solver_method, (20,10,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_sgd_3[i,j], temp    = rungraph(solver_method, (20,30,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_sgd_3[i,j], temp   = rungraph(solver_method, (10,5,3), X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_sgd_3[i,j], temp   = rungraph(solver_method, (30,20,10), X_train_pca, X_test_pca, y_train, y_test)
            
#            solver_method = 'NN_adam_1'
#            ac_NN3_adam_1[i,j], temp   = rungraph(solver_method, (20,10,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_adam_1[i,j], temp   = rungraph(solver_method, (20,30,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_adam_1[i,j], temp  = rungraph(solver_method, (10,5,3), X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_adam_1[i,j], temp  = rungraph(solver_method, (30,20,10), X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'NN_adam_2'
#            ac_NN3_adam_2[i,j], temp   = rungraph(solver_method, (20,10,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN5_adam_2[i,j], temp   = rungraph(solver_method, (20,30,5),  X_train_pca, X_test_pca, y_train, y_test)
#            ac_NN10_adam_2[i,j], temp  = rungraph(solver_method, (10,5,3), X_train_pca, X_test_pca, y_train, y_test)            
#            ac_NN20_adam_2[i,j], temp  = rungraph(solver_method, (30,20,10), X_train_pca, X_test_pca, y_train, y_test)
            solver_method = 'NN_adam_3'
            ac_NN3_adam_3[i,j], temp   = rungraph(solver_method, (100),  X_train_pca, X_test_pca, y_train, y_test)
            ac_NN5_adam_3[i,j], temp   = rungraph(solver_method, (10,10,10,10,10),  X_train_pca, X_test_pca, y_train, y_test)
            ac_NN10_adam_3[i,j], temp  = rungraph(solver_method, (50,50,50), X_train_pca, X_test_pca, y_train, y_test)            
            ac_NN20_adam_3[i,j], temp  = rungraph(solver_method, (30,30,30), X_train_pca, X_test_pca, y_train, y_test)
            
#            solver_method = 'SVM_linear'
#            ac_SVM_linear[i,j], temp  = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'SVM_poly_1'
#            ac_SVM_poly_1[i,j], temp  = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'SVM_poly_2'
#            ac_SVM_poly_2[i,j], temp  = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'SVM_poly_5'
#            ac_SVM_poly_5[i,j]  = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'SVM_rbf'
#            ac_SVM_rbf[i,j], temp     = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
#            solver_method = 'SVM_sigmoid'
#            ac_SVM_sigmoid[i,j] = rungraph(solver_method, 0,  X_train_pca, X_test_pca, y_train, y_test)
            
    # Print everything!
    printMAC('ac_NN3_lbfgs_1',ac_NN3_lbfgs_1)
    printMAC('ac_NN5_lbfgs_1',ac_NN5_lbfgs_1)
    printMAC('ac_NN10_lbfgs_1',ac_NN10_lbfgs_1)
    printMAC('ac_NN20_lbfgs_1',ac_NN20_lbfgs_1)
    printMAC('ac_NN3_lbfgs_2',ac_NN3_lbfgs_2)
    printMAC('ac_NN5_lbfgs_2',ac_NN5_lbfgs_2)
    printMAC('ac_NN10_lbfgs_2',ac_NN10_lbfgs_2)
    printMAC('ac_NN20_lbfgs_2',ac_NN20_lbfgs_2)
    printMAC('ac_NN3_lbfgs_3',ac_NN3_lbfgs_3)
    printMAC('ac_NN5_lbfgs_3',ac_NN5_lbfgs_3)
    printMAC('ac_NN10_lbfgs_3',ac_NN10_lbfgs_3)
    printMAC('ac_NN20_lbfgs_3',ac_NN20_lbfgs_3)
    
    printMAC('ac_NN3_sgd_1',ac_NN3_sgd_1)
    printMAC('ac_NN5_sgd_1',ac_NN5_sgd_1)
    printMAC('ac_NN10_sgd_1',ac_NN10_sgd_1)
    printMAC('ac_NN20_sgd_1',ac_NN20_sgd_1)
    printMAC('ac_NN3_sgd_2',ac_NN3_sgd_2)
    printMAC('ac_NN5_sgd_2',ac_NN5_sgd_2)
    printMAC('ac_NN10_sgd_2',ac_NN10_sgd_2)
    printMAC('ac_NN20_sgd_2',ac_NN20_sgd_2)
    printMAC('ac_NN3_sgd_3',ac_NN3_sgd_3)
    printMAC('ac_NN5_sgd_3',ac_NN5_sgd_3)
    printMAC('ac_NN10_sgd_3',ac_NN10_sgd_3)
    printMAC('ac_NN20_sgd_3',ac_NN20_sgd_3)
    
    printMAC('ac_NN3_adam_1',ac_NN3_adam_1)
    printMAC('ac_NN5_adam_1',ac_NN5_adam_1)
    printMAC('ac_NN10_adam_1',ac_NN10_adam_1)
    printMAC('ac_NN20_adam_1',ac_NN20_adam_1)
    printMAC('ac_NN3_adam_2',ac_NN3_adam_2)
    printMAC('ac_NN5_adam_2',ac_NN5_adam_2)
    printMAC('ac_NN10_adam_2',ac_NN10_adam_2)
    printMAC('ac_NN20_adam_2',ac_NN20_adam_2)
    printMAC('ac_NN3_adam_3',ac_NN3_adam_3)
    printMAC('ac_NN5_adam_3',ac_NN5_adam_3)
    printMAC('ac_NN10_adam_3',ac_NN10_adam_3)
    printMAC('ac_NN20_adam_3',ac_NN20_adam_3)

    printMAC('ac_SVM_linear',ac_SVM_linear)
    printMAC('ac_SVM_poly_1',ac_SVM_poly_1)
    printMAC('ac_SVM_poly_2',ac_SVM_poly_2)
    printMAC('ac_SVM_poly_5',ac_SVM_poly_5)
    printMAC('ac_SVM_rbf'   ,ac_SVM_rbf   )
    printMAC('ac_SVM_sigmoid',ac_SVM_sigmoid)
        
    print('THE END')

