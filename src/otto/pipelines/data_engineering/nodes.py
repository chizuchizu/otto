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

from typing import Any, Dict
import umap
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

from sklearn.decomposition import PCA
import cuml.manifold    as tsne_rapids
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


def docking(train: pd.DataFrame, test: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    print("docking")
    dict_a: Dict
    dict_a = {"Class_1": 0,
              "Class_2": 1,
              "Class_3": 2,
              "Class_4": 3,
              "Class_5": 4,
              "Class_6": 5,
              "Class_7": 6,
              "Class_8": 7,
              "Class_9": 8,
              }
    target = train["target"].map(dict_a)
    con_df = pd.concat([train, test], sort=False)
    """
    targetは Class_n　→ n-1　にした（1-9だったので）
    trainのcolumnsは feat_n → n にする
    """
    con_df = con_df.drop(["id", "target"], axis=1)
    con_df.columns = con_df.columns.map(lambda x: int(x[5:]))
    con_df = np.log1p(con_df)

    return con_df, target


def split_data(df, df_tsne, df_umap, df_pca, df_features, df_knn, target):
    # df = np.log1p(df)
    # df = pd.concat([df, df_pca], axis=1, join="inner")

    # df = pd.concat([df, df_pca], axis=1, join_axes=[df.index])
    df = pd.concat([df.reset_index(drop=True), df_tsne.reset_index(drop=True)], axis=1)
    df = pd.concat([df.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
    df = pd.concat([df.reset_index(drop=True), df_umap.reset_index(drop=True)], axis=1)
    # df = pd.concat([df.reset_index(drop=True), df_knn.reset_index(drop=True)], axis=1)

    # df = df_pca.copy()

    # df = pd.concat([df, df_features], axis=1, join_axes=[df.index])
    df = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    train_df = df[:len(target)]
    test_df = df[len(target):]
    df.to_csv("data/04_features/train2_csv", index=False)

    return train_df, test_df


def do_umap(df: pd.DataFrame, target: pd.Series, parameters: Dict):
    umap_path = parameters["umap_path"]

    if not os.path.isfile(umap_path):
        ump = umap.UMAP(n_components=2, n_neighbors=9)
        ump.fit(df.iloc[:len(target), :])
        embedding = ump.transform(df.iloc[:len(target), :])
        memo = ump.transform(df)
        np.save(umap_path, memo)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=target, alpha=0.4)
        plt.colorbar()
        plt.show()
    else:
        memo = np.load(umap_path)

    # print(embedding)
    return_df = pd.DataFrame(memo, columns=["umap1", "umap2"])
    return return_df


def do_pca(df: pd.DataFrame, target: pd.Series):
    n = 20
    pca = PCA(n_components=n)
    pca.fit(df[:len(target)])
    # df_pca = pca.fit_transform(df)
    df_pca = pca.transform(df)
    n_name = [f"pca{i}" for i in range(n)]
    df_pca = pd.DataFrame(df_pca, columns=n_name)
    return df_pca


def do_tSNE(df: pd.DataFrame, target: pd.Series):
    """
    https://www.kaggle.com/titericz/t-sne-visualization-with-rapids
    """
    tsne = tsne_rapids.TSNE(n_components=2, perplexity=30, verbose=2)
    # tsne.fit(df[:len(target)])
    df_tsne = tsne.fit_transform(df.values)
    n_name = [f"tsne{i}" for i in range(2)]

    # plt.scatter(df_tsne[:len(target), 0], df_tsne[:len(target), 1], c=target, s=0.5)
    # plt.show()
    return pd.DataFrame(df_tsne, columns=n_name)


def knn_train_model(
        df: pd.DataFrame, target: pd.DataFrame
):
    # n_splits = 5
    # folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #
    # oof = np.zeros((df.shape[0], 9))
    #
    # for trn_idx, val_idx in folds.split(df[:len(target)], target):
    #     train_x = df.iloc[trn_idx, :].values
    #     val_x = df.iloc[val_idx, :].values
    #     train_y = target[trn_idx].values
    #     val_y = target[val_idx].values
    #
    #     classifier = KNeighborsClassifier(n_neighbors=1024, n_jobs=12)
    #     classifier.fit(train_x, train_y)
    #
    #     y_hat = classifier.predict_proba(val_x)
    #
    #     print(log_loss(val_y, y_hat))
    #     print(oof.shape, y_hat.shape)
    #     oof[val_idx, :] = y_hat
    #     pred = classifier.predict_proba(df[len(target):])
    #
    #     oof[len(target):, :] = pred / n_splits
    #
    # print(oof.shape)
    # np.save("data/04_features/oof.npz", oof)
    oof = np.load("data/04_features/oof.npy")
    n_name = ["knn_{}".format(i)for i in range(9)]
    oof = pd.DataFrame(oof, columns=n_name)
    return oof


def make_features(df: pd.DataFrame):
    memo = pd.DataFrame()
    memo["count_zero"] = df[df == 0].count(axis=1)
    # memo["count_one"] = df[df == 1].count(axis=1)
    return memo
