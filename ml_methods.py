import constants

from sklearn import tree
from xgboost import XGBRegressor
import pandas as pd
from functools import partial


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
