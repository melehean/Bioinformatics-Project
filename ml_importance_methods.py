import constants

from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from xgboost import XGBRegressor
from mlxtend.evaluate import feature_importance_permutation
from sklearn.model_selection import cross_val_score
import numpy as np


def get_decision_tree_importance(chemical_data, chemical_groups, bacteria_data):
    decision_tree_object = tree.DecisionTreeRegressor(
        random_state=constants.SEED,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=2,
    )

    decision_tree_object.fit(chemical_data, bacteria_data)

    importances = feature_importance_permutation(
        X=chemical_data.to_numpy(),
        y=bacteria_data.to_numpy(),
        predict_method=decision_tree_object.predict,
        metric="r2",
        feature_groups=chemical_groups,
        seed=constants.SEED)

    r_squared = decision_tree_object.score(chemical_data, bacteria_data)
    cv_score = cross_val_score(
        decision_tree_object, chemical_data, bacteria_data)
    cv_score = np.mean(cv_score, axis=0)

    return importances, r_squared, cv_score


def get_xgboost_importance(chemical_data, chemical_groups, bacteria_data):
    xgboost_object = XGBRegressor(
        n_estimators=400,
        max_depth=2,
        eta=0.1,
        n_jobs=4,
        random_state=constants.SEED)

    xgboost_object.fit(chemical_data, bacteria_data)

    importances = feature_importance_permutation(
        X=chemical_data.to_numpy(),
        y=bacteria_data.to_numpy(),
        predict_method=xgboost_object.predict,
        metric="r2",
        feature_groups=chemical_groups,
        seed=constants.SEED)

    r_squared = xgboost_object.score(chemical_data, bacteria_data)
    cv_score = cross_val_score(
        xgboost_object, chemical_data, bacteria_data)
    cv_score = np.mean(cv_score, axis=0)

    return importances, r_squared, cv_score


def get_random_forest_importance(chemical_data, chemical_groups, bacteria_data):

    random_forest_object = RandomForestRegressor(
        n_estimators=100,
        bootstrap='True',
        n_jobs=-1,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=5,
        min_samples_split=10,
        oob_score=True,
        random_state=constants.SEED
    )
    random_forest_object.fit(chemical_data, bacteria_data)

    importances = feature_importance_permutation(
        X=chemical_data.to_numpy(),
        y=bacteria_data.to_numpy(),
        predict_method=random_forest_object.predict,
        metric="r2",
        feature_groups=chemical_groups,
        seed=constants.SEED)

    r_squared = random_forest_object.score(chemical_data, bacteria_data)
    # cv_score = cross_val_score(
    #     random_forest_object, chemical_data, bacteria_data)
    # cv_score = np.mean(cv_score)
    oob_score = random_forest_object.oob_score_

    return importances, r_squared, oob_score
