# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:58:30 2024

@author: user
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


def visualize_feature_importance(file_path):
    # 讀取資料
    data = pd.read_excel(file_path)  # 確保檔案路徑正確

    # 分離特徵和標籤
    X = data.drop(columns=['標籤', '商品_ID', '年_週','週銷售變化量', '數量', '單價'])
    y = data['標籤']

    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # 訓練隨機森林模型
    feature_names = X.columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    # 計算特徵重要性
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # 可視化特徵重要性
    forest_importances = pd.Series(importances, index=feature_names)

    # 設置中文字體
    plt.rcParams['font.family'] = 'Microsoft JhengHei'


    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
