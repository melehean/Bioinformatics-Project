import pandas

import data

import pandas as pd
from functools import partial


def find_important_relationships(importances, chemical_groups, threshold):
    important_relationships = []
    for index, importance in enumerate(importances):
        if importance > threshold:
            important_relationships.append(
                (chemical_groups[index], importances[index]))
    return important_relationships


def get_metrics_for_all_bacteria(chemical_data, chemical_groups, all_bacteria_data, function):
    data = all_bacteria_data
    all_metrics = []
    for _, row in data.iterrows():

        function_with_args = partial(
            function, chemical_data, chemical_groups, row)
        importances, r_squared, cv_score = function_with_args()

        all_metrics.append((importances, r_squared, cv_score))
    return all_metrics


def important_relationships_for_all_bacteria(chemical_data, all_bacteria_data, function, importance_threshold=0.3):
    correlated_groups = data.group_correlated_features(chemical_data)
    features_with_names = data.redefine_features(
        chemical_data.columns, correlated_groups)
    features_with_indices = data.redefine_features_with_indices(
        chemical_data, correlated_groups)

    all_metrics = get_metrics_for_all_bacteria(
        chemical_data, features_with_indices, all_bacteria_data, function)
    chemicals = []
    all_importances = []
    bacterias = []
    cv_scores = []
    R2_scores = []
    for i, (importances, score, cv_score) in enumerate(all_metrics):
        important_relationships = find_important_relationships(
            importances[0], features_with_names, importance_threshold)
        for relationship in important_relationships:
            chemicals.append(relationship[0])
            all_importances.append(relationship[1])
            cv_scores.append(cv_score)
            R2_scores.append(score)
            bacterias.append(i)
    df = pandas.DataFrame()
    df["Bacteria"] = bacterias
    df["Chemicals"] = chemicals
    df["Importance measure"] = all_importances
    df["CV score"] = cv_scores
    df["R-squared"] = R2_scores
    return df


def get_correlation(chemical_data, all_bacteria_data):
    df = pd.DataFrame(index=chemical_data.columns)
    for index, row in all_bacteria_data.iterrows():
        correlation = chemical_data.corrwith(row)
        correlation.rename(str(index), inplace=True)
        df = pd.concat([df, correlation], axis=1)
    return df
