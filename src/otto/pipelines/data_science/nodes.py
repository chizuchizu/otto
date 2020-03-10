# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from rgf.sklearn import RGFClassifier
import optuna
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape, LayerNormalization, PReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
# from keras_radam import RAdam
from tensorflow.keras.optimizers import SGD, Adam, Optimizer

from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
import shap


def xgb_train_model(
        df: pd.DataFrame, target: pd.DataFrame, test, parameters: Dict[str, Any]
):
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # test_x = df.iloc[len(target):, :]

    xgb_param = {
        "max_depth": parameters["max_depth"],
        "learning_rate": parameters["learning_rate"],
        # "n_jobs": parameters["n_jobs"],
        "objective": parameters["objective"],
        "subsample": 0.8,
        "min_child_weight": 5,
        "booster": "gbtree",
        "num_class": 9,
        "num_leaves": 64,
        "n_estimators": 1000
    }
    if parameters["gpu"]:
        xgb_param["device"] = "gpu"
        xgb_param["tree_method"] = "gpu_hist"
        xgb_param["max_bin"] = 32
    n_splits = 5
    n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        clf = xgb.XGBClassifier(**xgb_param)
        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_metric="mlogloss",
                verbose=200,
                early_stopping_rounds=500
                )

        # bst = xgb.train(params=xgb_param, dtrain=dtrain, evals=[(dtest, "eval")], early_stopping_rounds=300, num_boost_round=500,
        #                 verbose_eval=500)
        oof[val_idx] = clf.predict_proba(val_x)
        oof[len(target):] += clf.predict_proba(test.values) / n_splits

    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/xgb_{}.csv".format("2"))
    return oof[len(target):].values


def lgbm_train_model(
        train_x: pd.DataFrame, target: pd.DataFrame, test, parameters: Dict[str, Any]
):
    lgb_train = lgb.Dataset(train_x, target)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": 9,
        "max_depth": parameters["max_depth"],
        # "bagging_freq": 5,
        "learning_rate": parameters["learning_rate"],
        "metric": "multi_logloss",
        "num_leaves": 12,
        # "subsample": 0.7,
        "verbose": -1
    }

    lgbm_params = {
        'metric': 'multi_logloss', 'objective': 'multiclass', 'num_class': 9, 'verbosity': 1,
        'feature_fraction': 0.4, 'num_leaves': 139, 'bagging_fraction': 0.8254401463359962, 'bagging_freq': 3,
        'lambda_l1': 0.02563829140437355, 'lambda_l2': 9.594334397031103, 'min_child_samples': 100
    }

    extraction_cb = ModelExtractionCallback()
    callbacks = [
        extraction_cb,
    ]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gbdt = lgb.cv(lgbm_params,
                  lgb_train,
                  folds=folds,
                  early_stopping_rounds=300,
                  num_boost_round=2000,
                  verbose_eval=100,
                  callbacks=callbacks
                  )
    print("Best num_boost_round: ", len(gbdt["multi_logloss-mean"]))
    clf = lgb.train(lgbm_params, lgb_train, num_boost_round=len(gbdt["multi_logloss-mean"]), verbose_eval=100)
    print("DONE")

    boosters = extraction_cb.raw_boosters
    best_iteration = extraction_cb.best_iteration

    # Create oof prediction result
    fold_iter = folds.split(train_x, target)
    oof_preds = np.zeros((train_x.shape[0] + test.shape[0], 9))
    for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, boosters)):
        # print(val_idx)
        valid = train_x.iloc[val_idx, :].values
        oof_preds[val_idx, :] = booster.predict(valid, num_iteration=best_iteration)
        oof_preds[len(target):, :] += booster.predict(test.values, num_iteration=best_iteration) / 5
        print(n_fold, "DONE")
    oof = pd.DataFrame(oof_preds)
    oof.to_csv("data/09_oof/lgbm_{}.csv".format("4"))

    clf.save_model(f"data/06_models/lgb_{datetime.today()}.txt")

    lgb.plot_importance(clf, max_num_features=20, importance_type="gain")
    plt.show()
    return clf


def nn_train_model(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict[str, Any]
):
    """
    Tensor Boardの見方

    %load_ext tensorboard

    """
    n_splits = 5
    num_class = 9
    epochs = 20
    lr_init = 0.01
    bs = 256
    num_features = df.shape[1]
    folds = StratifiedKFold(n_splits=n_splits, random_state=71, shuffle=True)

    def lr_scheduler(epoch):
        if epoch <= epochs * 0.8:
            return lr_init
        else:
            return lr_init * 0.1

    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),

        Dense(2 ** 10, kernel_initializer='glorot_uniform'),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 9, kernel_initializer='glorot_uniform', ),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 7, kernel_initializer='glorot_uniform'),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 6, kernel_initializer='glorot_uniform'),
        PReLU(),
        BatchNormalization(),
        Dropout(0.25),

        # ｒ
        # Dense(64, kernel_initializer='glorot_uniform', activation="relu"),
        # BatchNormalization(),
        # Dropout(0.25),

        Dense(num_class, activation="softmax")
    ])

    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(lr=lr_init, decay=0.0001)
    # optimizer = SGD(learning_rate=lr_init)

    init_weights1 = model.get_weights()

    """callbacks"""
    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    # callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda ep: float(lr_init / 3 ** (ep * 4 // epochs))))

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'))

    print(
        "\nIf you want to watch TF Board, you should enter the command."
        "\n%load_ext tensorboard\n%tensorboard --logdir {}\n".format(log_dir))

    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    preds = np.zeros((test.shape[0], num_class))
    oof = np.zeros((df.shape[0] + test.shape[0], num_class))
    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        # train_x = np.reshape(train_x, (-1, num_features, 1))
        # val_x = np.reshape(val_x, (-1, num_features, 1))
        model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2, batch_size=bs,
                  callbacks=callbacks)
        # preds += model.predict(test.values) / n_splits
        oof[val_idx] = model.predict(val_x)
        oof[len(target):] += model.predict(test.values) / n_splits
        model.set_weights(init_weights1)

    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/nn_{}.csv".format("normal_7"))

    print(
        "\nIf you want to watch TF Board, you should enter the command."
        "\n%load_ext tensorboard\n%tensorboard --logdir {}\n".format(log_dir))

    return oof[len(target):].values


def knn_train_model(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict
):
    n_splits = 5
    n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=14)
        classifier.fit(train_x, train_y)

        y_hat = classifier.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx, :] = y_hat
        pred = classifier.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    # oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i) for i in range(9)]
    oof = pd.DataFrame(oof, columns=n_name)
    oof.to_csv("data/09_oof/knn_{}.csv".format(n_neighbors))
    return oof[len(target):].values


def Ada(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict
):
    n_splits = 5
    # n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        classifier = AdaBoostClassifier(n_estimators=50,
                                        learning_rate=1,
                                        algorithm="SAMME.R",
                                        random_state=12345)
        classifier.fit(train_x, train_y)

        y_hat = classifier.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx] = y_hat
        pred = classifier.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    # oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i) for i in range(9)]
    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/ada_{}.csv".format("2"))
    return oof[len(target):].values


def extratrees(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict
):
    n_splits = 5
    # n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        classifier = ExtraTreesClassifier(n_jobs=14, n_estimators=100, max_depth=10)
        classifier.fit(train_x, train_y)

        y_hat = classifier.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx] = y_hat
        pred = classifier.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    # oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i) for i in range(9)]
    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/extra_{}.csv".format("5"))
    return oof[len(target):].values


def rf(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict
):
    n_splits = 5
    # n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        classifier = RandomForestClassifier(n_jobs=14, n_estimators=1000,
                                            min_samples_leaf=1,
                                            max_features="auto",
                                            criterion="gini")
        classifier.fit(train_x, train_y)

        y_hat = classifier.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx] = y_hat
        pred = classifier.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    # oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i) for i in range(9)]
    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/rf_{}.csv".format("2"))
    return oof[len(target):].values


def rgf(
        df: pd.DataFrame, target: pd.DataFrame, test: pd.DataFrame, parameters: Dict
):
    n_splits = 5
    # n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((df.shape[0] + test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        classifier = RGFClassifier(n_jobs=14,
                                   algorithm="RGF",
                                   loss="Log",
                                   )
        classifier.fit(train_x, train_y)

        y_hat = classifier.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx] = y_hat
        pred = classifier.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    # oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i) for i in range(9)]
    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/rgf_{}.csv".format(3))
    return oof[len(target):].values


def predict(model, test_x) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    pred = model.predict(test_x)
    # Return the index of the class with max probability for all samples
    return pred


def stacking(train, target, test, parameters):
    path = "data/09_oof/"
    df_train = train.copy()
    df_test = test.copy()
    for x in parameters["model_list"]:
        oof_data = pd.DataFrame(pd.read_csv(path + x + ".csv").iloc[:, 1:].values)
        df_train = pd.concat([df_train.reset_index(drop=True), oof_data[:len(target)].reset_index(drop=True)], axis=1)
        df_test = pd.concat([df_test.reset_index(drop=True), oof_data[len(target):].reset_index(drop=True)], axis=1)

    lgb_train = lgb.Dataset(df_train, target)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": 9,
        "max_depth": -1,
        # "bagging_freq": 5,
        "learning_rate": 0.04,
        "feature_fraction": 0.3,
        "metric": "multi_logloss",
        "num_leaves": 12,
        # "subsample": 0.7,
        "verbose": -1
    }
    lgbm_params = {
        'bagging_freq': 5, 'bagging_fraction': 0.8, 'boost_from_average': 'false', 'boost': 'gbdt',
        'feature_fraction': 0.4, 'learning_rate': 0.03, 'max_depth': -1, 'metrics': 'multi_logloss',
        'min_data_in_leaf': 10, 'min_sum_hessian_in_leaf': 8.0, 'num_leaves': 8, 'num_threads': 8,
        'tree_learner': 'serial', 'objective': 'multiclass', 'num_class': 9, 'verbosity': -1
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gbdt = lgb.cv(lgbm_params,
                  lgb_train,
                  folds=folds,
                  early_stopping_rounds=200,
                  num_boost_round=2000,
                  verbose_eval=100,
                  )
    print("Best num_boost_round: ", len(gbdt["multi_logloss-mean"]))
    clf = lgb.train(lgbm_params, lgb_train, num_boost_round=len(gbdt["multi_logloss-mean"]), verbose_eval=100)

    test_pred = clf.predict(df_test)
    print("DONE")
    return test_pred


def stacking_NN(train, target, test, parameters):
    n_splits = 5
    num_class = 9
    epochs = 20
    lr_init = 0.01
    bs = 256
    folds = StratifiedKFold(n_splits=n_splits, random_state=71, shuffle=True)

    path = "data/09_oof/"
    df_train = train.copy()
    df_test = test.copy()
    for x in parameters["model_list"]:
        oof_data = pd.DataFrame(pd.read_csv(path + x + ".csv").iloc[:, 1:].values)
        df_train = pd.concat([df_train.reset_index(drop=True), oof_data[:len(target)].reset_index(drop=True)], axis=1)
        df_test = pd.concat([df_test.reset_index(drop=True), oof_data[len(target):].reset_index(drop=True)], axis=1)

    num_features = df_train.shape[1]

    def lr_scheduler(epoch):
        if epoch <= epochs * 0.8:
            return lr_init
        else:
            return lr_init * 0.1

    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),

        Dense(2 ** 10, kernel_initializer='glorot_uniform'),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 9, kernel_initializer='glorot_uniform', ),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 7, kernel_initializer='glorot_uniform'),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2 ** 6, kernel_initializer='glorot_uniform'),
        PReLU(),
        BatchNormalization(),
        Dropout(0.25),

        # ｒ
        # Dense(64, kernel_initializer='glorot_uniform', activation="relu"),
        # BatchNormalization(),
        # Dropout(0.25),

        Dense(num_class, activation="softmax")
    ])

    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(lr=lr_init, decay=0.0001)
    # optimizer = SGD(learning_rate=lr_init)

    init_weights1 = model.get_weights()

    """callbacks"""
    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    # callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda ep: float(lr_init / 3 ** (ep * 4 // epochs))))

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'))

    print(
        "\nIf you want to watch TF Board, you should enter the command."
        "\n%load_ext tensorboard\n%tensorboard --logdir {}\n".format(log_dir))

    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    preds = np.zeros((df_test.shape[0], num_class))
    # oof = np.zeros((train.shape[0] + test.shape[0], num_class))
    for trn_idx, val_idx in folds.split(df_train, target):
        train_x = df_train.iloc[trn_idx, :].values
        val_x = df_train.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        # train_x = np.reshape(train_x, (-1, num_features, 1))
        # val_x = np.reshape(val_x, (-1, num_features, 1))
        model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2, batch_size=bs,
                  callbacks=callbacks)
        # preds += model.predict(test.values) / n_splits
        # oof[val_idx] = model.predict(val_x)
        # oof[len(target):] += model.predict(test.values) / n_splits
        preds += model.predict(df_test.values) / n_splits
        model.set_weights(init_weights1)

    return preds


def cat_boost(
        train, target, test, parameters
):
    n_splits = 5
    # n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros((train.shape[0] + test.shape[0], 9))

    model = CatBoostClassifier(
        # learning_rate=0.01,
        depth=11,
        task_type="GPU",
        loss_function="MultiClass",
        verbose=200,
        l2_leaf_reg=1,
        random_strength=2,
        bagging_temperature=0.6499735309792984,
        iterations=440,
        od_wait=45,
        # od_type="incToDec"
    )
    params = {'iterations': 440, 'depth': 11, 'l2_leaf_reg': 1, 'random_strength': 2,
              'bagging_temperature': 0.6499735309792984, 'od_type': 'IncToDec', 'od_wait': 45}

    for trn_idx, val_idx in folds.split(train, target):
        train_x = train.iloc[trn_idx, :].values
        val_x = train.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        train_pool = Pool(data=train_x, label=train_y)
        val_pool = Pool(data=val_x, label=val_y)

        model.fit(train_pool, eval_set=val_pool)

        y_hat = model.predict_proba(val_x)

        print(log_loss(val_y, y_hat))
        print(oof.shape, y_hat.shape)
        oof[val_idx] = y_hat
        pred = model.predict_proba(test.values)

        oof[len(target):, :] += pred / n_splits

    oof = pd.DataFrame(oof)
    oof.to_csv("data/09_oof/cb_{}.csv".format(2))
    return oof[len(target):]


def catb_optuna(
        train, target, test, parameters
):
    train_x, test_x, train_y, test_y = train_test_split(train.values, target.values, test_size=0.2, random_state=42)
    train_pool = Pool(data=train_x, label=train_y, cat_features=[])
    test_pool = Pool(data=test_x, label=test_y, cat_features=[])

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 6, 12),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 100),
            # 'learning_rate': 0.03,
            "task_type": "GPU",
            "loss_function": "MultiClass",
            'random_strength': trial.suggest_int('random_strength', 0, 100),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            "verbose": 500
        }
        model = CatBoostClassifier(**params)
        model.fit(train_pool)

        preds = model.predict_proba(test_pool)
        loss = log_loss(test_y, preds)
        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_trial)


def stacking_xgb(
        df: pd.DataFrame, target: pd.DataFrame, test, parameters: Dict[str, Any]
):
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # test_x = df.iloc[len(target):, :]
    path = "data/09_oof/"
    df_train = df.copy()
    df_test = test.copy()
    for x in parameters["model_list"]:
        oof_data = pd.DataFrame(pd.read_csv(path + x + ".csv").iloc[:, 1:].values)
        df_train = pd.concat([df_train.reset_index(drop=True), oof_data[:len(target)].reset_index(drop=True)], axis=1)
        df_test = pd.concat([df_test.reset_index(drop=True), oof_data[len(target):].reset_index(drop=True)], axis=1)


    xgb_param = {
        "max_depth": parameters["max_depth"],
        "learning_rate": parameters["learning_rate"],
        # "n_jobs": parameters["n_jobs"],
        "objective": parameters["objective"],
        "subsample": 0.8,
        "min_child_weight": 5,
        "booster": "gbtree",
        "num_class": 9,
        "num_leaves": 64,
        "n_estimators": 1000
    }
    if parameters["gpu"]:
        xgb_param["device"] = "gpu"
        xgb_param["tree_method"] = "gpu_hist"
        xgb_param["max_bin"] = 32
    n_splits = 5
    n_neighbors = parameters["n_neighbors"]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # oof = np.zeros((df.shape[0] + test.shape[0], 9))
    pred = np.zeros((test.shape[0], 9))

    for trn_idx, val_idx in folds.split(df_train, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        clf = xgb.XGBClassifier(**xgb_param)
        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_metric="mlogloss",
                verbose=200,
                early_stopping_rounds=500
                )

        # bst = xgb.train(params=xgb_param, dtrain=dtrain, evals=[(dtest, "eval")], early_stopping_rounds=300, num_boost_round=500,
        #                 verbose_eval=500)
        pred += clf.predict_proba(df_test.values) / n_splits

    return pred


def make_submit_file(pred: np.ndarray, ss: pd.DataFrame) -> None:
    save_path = "data/07_model_output/{}.csv".format(datetime.today())
    ss.iloc[:, 1:] = pred
    ss.to_csv(save_path, index=None)


def features_importance(model, df: pd.DataFrame):
    importance = pd.DataFrame(model.features_importance(importance_type="gain"), index=df.columns,
                              columns=["importance"])
    print(importance.head(20))


class ModelExtractionCallback(object):
    """Callback class for retrieving trained model from lightgbm.cv()
    NOTE: This class depends on '_CVBooster' which is hidden class, so it might doesn't work if the specification is changed.
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # Saving _CVBooster object.
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # Throw exception if the callback class is not called.
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # return Booster object
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # return list of Booster
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # return boosting round when early stopping.
        return self._model.best_iteration
