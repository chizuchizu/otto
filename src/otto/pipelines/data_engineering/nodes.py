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
import scipy as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cuml.manifold    as tsne_rapids
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    con_df, ignore = count_feature(con_df)
    con_df = np.log1p(con_df)

    con_df = pd.DataFrame(RI(con_df, 140, 2, normalize=False, seed=10))

    return con_df, target


def split_data(df, df_tsne, df_umap, df_pca, df_features, df_kmeans, target):
    # df = np.log1p(df)
    # df = pd.concat([df, df_pca], axis=1, join="inner")

    # df = pd.concat([df, df_pca], axis=1, join_axes=[df.index])

    df = pd.concat([df.reset_index(drop=True), df_tsne.reset_index(drop=True)], axis=1)
    df = pd.concat([df.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
    df = pd.concat([df.reset_index(drop=True), df_umap.reset_index(drop=True)], axis=1)
    df = pd.concat([df.reset_index(drop=True), df_kmeans.reset_index(drop=True)], axis=1)

    # df = df_pca.copy()

    # df = pd.concat([df, df_features], axis=1, join_axes=[df.index])
    df = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

    R = col_k_ones_matrix(df.shape[1], 130, 2, 4, seed=101)
    np.random.seed(111)
    R.data = np.random.choice([1, -1], R.data.size)
    X3 = df.values * R
    X1 = np.sign(X3) * np.abs(X3) ** .6
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(X1))

    train_df = df[:len(target)]
    test_df = df[len(target):]
    # df.to_csv("data/04_features/train2_csv", index=False)

    oof_path = datetime.today()

    return train_df, test_df, oof_path


def do_kmeans(df: pd.DataFrame, target: pd.Series):
    kmeans = KMeans(n_clusters=2, n_jobs=13, random_state=42)
    kmeans.fit(df[:len(target)].values)
    memo = kmeans.transform(df.values)
    return pd.DataFrame(memo)


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
    tsne = tsne_rapids.TSNE(n_components=2, perplexity=20, verbose=2)
    # tsne.fit(df[:len(target)])
    df_tsne = tsne.fit_transform(df.values)
    n_name = [f"tsne{i}" for i in range(2)]

    # plt.scatter(df_tsne[:len(target), 0], df_tsne[:len(target), 1], c=target, s=0.5)
    # plt.show()
    return pd.DataFrame(df_tsne, columns=n_name)


def make_features(df: pd.DataFrame):
    memo = pd.DataFrame()
    memo["count_zero"] = df[df == 0].count(axis=1)
    memo["count_one"] = df[df == 1].count(axis=1)
    memo["count_two"] = df[df == 2].count(axis=1)

    memo["max_val"] = df.sum(axis=1)
    memo["sum_val"] = df.sum(axis=1)
    # memo["count_one"] = df[df == 1].count(axis=1)
    return memo


# https://github.com/tks0123456789/kaggle-Otto/blob/abf9532846950b32b2b8f0ec03fd272f33e4a761/utility.py#L5


def count_feature(X, tbl_lst=None, min_cnt=1):
    X_lst = [pd.Series(X.iloc[:, i]) for i in range(X.shape[1])]
    if tbl_lst is None:
        tbl_lst = [x.value_counts() for x in X_lst]
        if min_cnt > 1:
            tbl_lst = [s[s >= min_cnt] for s in tbl_lst]
    X = sp.column_stack([x.map(tbl).values for x, tbl in zip(X_lst, tbl_lst)])
    # NA(unseen values) to 0
    return np.nan_to_num(X), tbl_lst


# mat: A sparse matrix
def remove_duplicate_cols(mat):
    if not isinstance(mat, sp.sparse.coo_matrix):
        mat = mat.tocoo()
    row = mat.row
    col = mat.col
    data = mat.data
    crd = pd.DataFrame({'row': row, 'col': col, 'data': data}, columns=['col', 'row', 'data'])
    col_rd = crd.groupby('col').apply(lambda x: str(np.array(x)[:, 1:]))
    dup = col_rd.duplicated()
    return mat.tocsc()[:, col_rd.index.values[dup.values == False]]


def RImatrix(p, m, k, rm_dup_cols=False, seed=None):
    """ USAGE:
    Argument
      p: # of original varables
      m: The length of index vector
      k: # of 1s == # of -1s
    Rerurn value
      sparce.coo_matrix, shape:(p, s)
      If rm_dup_cols == False s == m
      else s <= m
    """
    if seed is not None: np.random.seed(seed)
    popu = range(m)
    row = np.repeat(range(p), 2 * k)  # 繰り返す
    col = np.array([np.random.choice(popu, 2 * k, replace=False) for i in range(p)]).reshape((p * k * 2,))  # n*k, 2
    data = np.tile(np.repeat([1, -1], k), p)
    # 疎行列を利用している
    mat = sp.sparse.coo_matrix((data, (row, col)), shape=(p, m), dtype=sp.int8)
    if rm_dup_cols:
        mat = remove_duplicate_cols(mat)
    return mat


# Random Indexing
def RI(X, m, k=1, normalize=True, seed=None, returnR=False):
    R = RImatrix(X.shape[1], m, k, rm_dup_cols=True, seed=seed)
    Mat = X * R
    if normalize:
        Mat = pd.normalize(Mat, norm='l2')
    if returnR:
        return Mat, R
    else:
        return Mat


# Return a sparse matrix whose column has k_min to k_max 1s
def col_k_ones_matrix(p, m, k=None, k_min=1, k_max=1, seed=None, rm_dup_cols=True):
    if k is not None:
        k_min = k_max = k
    if seed is not None: np.random.seed(seed)
    k_col = np.random.choice(range(k_min, k_max + 1), m)
    col = np.repeat(range(m), k_col)
    popu = np.arange(p)
    l = [np.random.choice(popu, k_col[i], replace=False).tolist() for i in range(m)]
    row = sum(l, [])
    data = np.ones(k_col.sum())
    mat = sp.sparse.coo_matrix((data, (row, col)), shape=(p, m), dtype=np.float32)
    if rm_dup_cols:
        mat = remove_duplicate_cols(mat)
    return mat