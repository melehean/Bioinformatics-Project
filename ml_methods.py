import pandas

import constants
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn import tree
from xgboost import XGBRegressor
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import rfpimp
import itertools


from sklearn.feature_selection import SelectKBest, f_regression
def get_decision_tree_importance(chemical_data, bacteria_data):
    decision_tree_object = tree.DecisionTreeRegressor(
        random_state=constants.SEED)

    decision_tree_object.fit(chemical_data, bacteria_data)
    importance = pd.DataFrame(data=decision_tree_object.feature_importances_,
                              index=chemical_data.columns)
    return importance.idxmax().values[0], importance.max().values[0]


def get_correlation(chemical_data, bacteria_data):
    correlation = chemical_data.corrwith(bacteria_data)
    best_chemical = correlation.abs().idxmax()
    return best_chemical, correlation[best_chemical]


def get_xgboost_importance(chemical_data, bacteria_data):
    xgboost_object = XGBRegressor(random_state=constants.SEED)
    xgboost_object.fit(chemical_data, bacteria_data)
    importance = pd.DataFrame(data=xgboost_object.feature_importances_,
                              index=chemical_data.columns)
    return importance.idxmax().values[0], importance.max().values[0]

def random_forest_importance(chemical_data, bacteria_data):
    forest = RandomForestRegressor(
        n_estimators=100,
        bootstrap='True',
        n_jobs=-1,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=5,
        min_samples_split=10,
        oob_score=True,
        random_state=42
    )
    #model_leave_one_out(forest, chemical_data, bacteria_data)
    forest.fit(chemical_data, bacteria_data)
    # importance = pd.Series(data=forest.feature_importances_,
    #                           index=chemical_data.columns)
    # importance = importance.sort_values(ascending=True)
    r2 = forest.score(chemical_data, bacteria_data)
    oob = forest.oob_score_
    print("Forest OOB scorem {}".format(oob))
    print("Forest R2 score {}".format(r2))
    correlated_groups = group_correlated_features(chemical_data)
    features = redefine_features(chemical_data.columns, correlated_groups)
    importances = rfpimp.importances(forest, chemical_data, bacteria_data, features=features)
    return importances, forest

def model_leave_one_out(model, X, y):
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    min_score, min_index = min(scores), np.argmin(scores)
    avg_score = np.mean(scores)
    print("Crossvalidation_results ...")
    print("Min score {}".format(min_score))
    print("Average score {}".format(avg_score))
    return scores

def get_results_for_all_bacteria(chemical_data, all_bacteria_data, function):
    all_metric_value = []
    all_chemical = []
    for _, row in all_bacteria_data.iterrows():
        function_with_args = partial(function, chemical_data, row)
        chemical, metric_value = function_with_args()
        all_metric_value.append(metric_value)
        all_chemical.append(chemical)

    df = pd.DataFrame()
    df["Chemicals"] = all_chemical
    df["Metric"] = all_metric_value
    return df

def plot_importance_for_specific_bacteria(chemical_data, bacteria, function, top=5):
    function_with_args = partial(function, chemical_data, bacteria)
    importances, models = function_with_args()
    importances[:top].plot(kind = 'barh')

def group_correlated_features(chemical_data, abs_threshold=0.5):
    features_corr = chemical_data.corr()
    mask = features_corr.abs() > abs_threshold
    groups = []
    for column in mask.columns:
        correlated_items = [column]
        correlated_with_any = False
        group_existent = False
        for items in mask[column].items():
            if items[0] == column:
                continue
            elif items[1] == True:
                correlated_items.append(items[0])
                correlated_with_any = True
        group_existent = set(correlated_items) in [set(group) for group in groups]
        if correlated_with_any and not group_existent:
            groups.append(correlated_items)
    return groups

def redefine_features(features, groups):
    for feature in features:
        if not any(feature in group for group in groups):
            groups.append(feature)
    return groups
def features_corr_matrix(chemical_data):
    #f, ax = plt.subplots(figsize=(10, 8))
    corr = chemical_data.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)

def find_important_relationships(importances, threshold):
    important_relationships = []
    for group, importance in importances.iterrows():
        importance = importance[0]
        if importance > threshold:
            important_relationships.append((group, importance))
        else:
            break
    return important_relationships

def get_metrics_for_all_bacteria(chemical_data, all_bacteria_data, function, use_first=None):
    if use_first is not None:
        data = all_bacteria_data.iloc[:use_first]
    else:
        data = all_bacteria_data
    all_metrics = []
    for _, row in data.iterrows():
        function_with_args = partial(function, chemical_data, row)
        importances, forest = function_with_args()
        all_metrics.append((importances, forest.score(chemical_data, row), forest.oob_score_))
    return all_metrics

def important_relationships_for_all_bacteria(chemical_data, all_bacteria_data, function, importance_threshold = 0.3, use_first=None):
    all_metrics = get_metrics_for_all_bacteria(chemical_data, all_bacteria_data, function, use_first)
    chemicals = []
    all_importances = []
    bacterias = []
    oob_scores = []
    R2_scores = []
    for i, (importances, score, oob_score) in enumerate(all_metrics):
        important_relationships = find_important_relationships(importances, importance_threshold)
        for relationship in important_relationships:
            chemicals.append(relationship[0])
            all_importances.append(relationship[1])
            oob_scores.append(oob_score)
            R2_scores.append(score)
            bacterias.append(i)
    df = pandas.DataFrame()
    df["Bacteria"] = bacterias
    df["Chemicals"] = chemicals
    df["Importance measure"] = all_importances
    df["OOB score"] = oob_scores
    df["R-squared"] = R2_scores
    return df