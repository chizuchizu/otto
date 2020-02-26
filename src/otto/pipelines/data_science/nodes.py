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
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
import xgboost as xgb


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
        "max_depth": 14,
        "learning_rate": parameters["learning_rate"],
        "metric": "multi_logloss",
        "num_leaves": 50
    }
    gbdt = lgb.cv(lgbm_params,
                  lgb_train,
                  nfold=5,
                  early_stopping_rounds=100,
                  num_boost_round=1000,
                  verbose_eval=100
                  )
    print("Best num_boost_round: ", len(gbdt["multi_logloss-mean"]))
    clf = lgb.train(lgbm_params, lgb_train, num_boost_round=len(gbdt["multi_logloss-mean"]), verbose_eval=100)
    print("DONE")
    clf.save_model(f"data/06_models/lgb_{datetime.today()}.txt")

    lgb.plot_importance(clf, max_num_features=20, importance_type="gain")
    plt.show()
    return clf


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
