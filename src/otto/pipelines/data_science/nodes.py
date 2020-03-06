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
from sklearn.decomposition import PCA

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape, LayerNormalization, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dropout

from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
import xgboost as xgb
import shap


def xgb_train_model(
        df: pd.DataFrame, target: pd.DataFrame, parameters: Dict[str, Any]
):
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    train_x = df.iloc[:len(target), :]
    # test_x = df.iloc[len(target):, :]

    xgtrain = xgb.DMatrix(train_x, label=target)

    xgb_param = {
        "max_depth": parameters["max_depth"],
        "learning_rate": parameters["learning_rate"],
        # "n_jobs": parameters["n_jobs"],
        "objective": parameters["objective"],
        "booster": "gbtree",
        "num_class": 9,
        "num_leaves": 64
    }
    if parameters["gpu"]:
        xgb_param["device"] = "gpu"
        xgb_param["tree_method"] = "gpu_hist"
        xgb_param["max_bin"] = 1024

    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=500, nfold=5, metrics=parameters["metric"],
                      early_stopping_rounds=100, verbose_eval=50)
    print("Best number of trees = {}".format(cvresult.shape[0]))
    gbdt = xgb.train(xgb_param, xgtrain, num_boost_round=cvresult.shape[0])
    return gbdt


def lgbm_train_model(
        train_x: pd.DataFrame, target: pd.DataFrame, parameters: Dict[str, Any]
):
    lgb_train = lgb.Dataset(train_x, target)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": 9,
        "max_depth": parameters["max_depth"],
        # "bagging_freq": 5,
        "learning_rate": parameters["learning_rate"],
        "metric": "multi_logloss",
        #         "num_leaves": 16,
        # "subsample": 0.7,
        "verbose": -1
    }
    gbdt = lgb.cv(lgbm_params,
                  lgb_train,
                  nfold=5,
                  early_stopping_rounds=100,
                  num_boost_round=2000,
                  verbose_eval=100
                  )
    print("Best num_boost_round: ", len(gbdt["multi_logloss-mean"]))
    clf = lgb.train(lgbm_params, lgb_train, num_boost_round=len(gbdt["multi_logloss-mean"]), verbose_eval=100)
    print("DONE")
    clf.save_model(f"data/06_models/lgb_{datetime.today()}.txt")

    lgb.plot_importance(clf, max_num_features=20, importance_type="gain")
    plt.show()
    return clf


def cnn_train_model(
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
    folds = KFold(n_splits=n_splits, random_state=71, shuffle=True)

    def lr_scheduler(epoch):
        if epoch <= epochs * 0.8:
            return lr_init
        else:
            return lr_init * 0.1

    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(1024, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(512, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(256, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(128, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(128, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, kernel_initializer='glorot_uniform', activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(num_class, activation="softmax")
    ])

    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(lr=lr_init, decay=0.0001)

    """callbacks"""
    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    preds = np.zeros((test.shape[0], num_class))
    for trn_idx, val_idx in folds.split(df, target):
        train_x = df.iloc[trn_idx, :].values
        val_x = df.iloc[val_idx, :].values
        train_y = target[trn_idx].values
        val_y = target[val_idx].values

        # train_x = np.reshape(train_x, (-1, num_features, 1))
        # val_x = np.reshape(val_x, (-1, num_features, 1))
        model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, verbose=2, batch_size=bs,
                  callbacks=callbacks)
        preds += model.predict(test.values) / n_splits
    print(
        "\nIf you want to watch TF Board, you should enter the command."
        "\n%load_ext tensorboard\n%tensorboard --logdir {}\n".format(log_dir))

    return preds


def get_features(preds, df):
    return None


def predict(model, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    pred = model.predict(test_x)
    # Return the index of the class with max probability for all samples
    return pred


def make_submit_file(pred: np.ndarray, ss: pd.DataFrame) -> None:
    save_path = "data/07_model_output/{}.csv".format(datetime.today())
    ss.iloc[:, 1:] = pred
    ss.to_csv(save_path, index=None)


def features_importance(model, df: pd.DataFrame):
    importance = pd.DataFrame(model.features_importance(importance_type="gain"), index=df.columns,
                              columns=["importance"])
    print(importance.head(20))
