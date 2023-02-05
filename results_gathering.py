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


def important_relationships_for_all_bacteria(chemical_data, all_bacteria_data, function, model_name, importance_threshold=0.3):
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
    df[f"Importance_{model_name}"] = all_importances
    df[f"CV_{model_name}"] = cv_scores
    df[f"R_squared_{model_name}"] = R2_scores
    return df


def prepare_final_results(
    chemical_data,
    bacteria_data,
    decision_tree_importance,
    xgboost_importance,
    random_forest_importance
):
    # Merge three importance df into one
    random_forest_importance["Chemicals"] = random_forest_importance["Chemicals"].astype(
        str)
    xgboost_importance["Chemicals"] = xgboost_importance["Chemicals"].astype(
        str)
    decision_tree_importance["Chemicals"] = decision_tree_importance["Chemicals"].astype(
        str)
    df_merged = random_forest_importance.merge(
        xgboost_importance, on=["Bacteria", "Chemicals"])
    df_merged = df_merged.merge(decision_tree_importance, on=[
                                "Bacteria", "Chemicals"])

    # Count bacteria for all common bacteria
    bacteria_counts = []
    for bacteria_index in df_merged["Bacteria"]:
        bacteria_counts.append(
            bacteria_data.iloc[[bacteria_index]].sum(axis=1).values[0])
    df_merged["Bacteria_counts"] = bacteria_counts
    df_merged = df_merged.iloc[:, [0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    # Calculate correlation for all common bacteria and best chemicals
    correlations = []
    for _, row in df_merged.iterrows():
        bacteria_index = row["Bacteria"]
        bacteria_row = bacteria_data.iloc[bacteria_index]
        chemical_list_string = row["Chemicals"]
        proper_chemical_list = chemical_list_string.strip('][').replace('\'', '').split(', ')
        chemical_data_for_bacteria = chemical_data[proper_chemical_list]
        correlations.append(
            chemical_data_for_bacteria.corrwith(bacteria_row).values[0])

    df_merged["Correlation"] = correlations
    df_merged = df_merged.iloc[:, [0, 1, 2, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    return df_merged
