import ml_methods
import data
from IPython.display import display
import constants
import matplotlib.pyplot as plt

chemical_data = data.prepare_chemical_data()
type_data = data.prepare_type_data()
# chemical_data.to_csv("chemical_data.csv")
#type_data.to_csv("bacteria_data.csv")
# bacteria_index = 46
# ml_methods.plot_importance_for_specific_bacteria(chemical_data, type_data.iloc[bacteria_index], ml_methods.random_forest_importance, 5)
# plt.show()
# ml_methods.features_corr_matrix(chemical_data)
# plt.show()
# groups = ml_methods.group_correlated_features(chemical_data)
# print(groups)
# ml_methods.random_forest_importance(chemical_data, type_data.iloc[bacteria_index])
relationships = ml_methods.important_relationships_for_all_bacteria(chemical_data, type_data, ml_methods.random_forest_importance, use_first=None ,importance_threshold=0.20)
display(relationships.sort_values('Importance measure', ascending=False))
relationships.to_csv("relationships.csv")