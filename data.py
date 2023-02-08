import constants

import pandas as pd
import seaborn as sns
import numpy as np
from copy import deepcopy
from IPython.display import display

def draw_chemical_correlation_matrix(chemical_data):
    corr = chemical_data.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)


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
        group_existent = set(correlated_items) in [
            set(group) for group in groups]
        if correlated_with_any and not group_existent:
            groups.append(correlated_items)
    return groups


def redefine_features(features, groups):
    final_groups = deepcopy(groups)
    for feature in features:
        if not any(feature in group for group in groups):
            final_groups.append(feature)
    return final_groups


def redefine_features_with_indices(chemical_data, groups):
    features = chemical_data.columns.to_list()

    indices_group = deepcopy(groups)
    for array_index, feature_array in enumerate(groups):
        for index, feature in enumerate(feature_array):
            indices_group[array_index][index] = features.index(feature)

    final_indices_group = deepcopy(indices_group)
    for index, _ in enumerate(features):
        flag = False
        for group in indices_group:
            if index in group:
                flag = True
                break
        if not flag:
            final_indices_group.append(index)
    return final_indices_group


def prepare_chemical_data():
    chemical_data = pd.read_csv(constants.CHEMICAL_DATA_PATH)
    chemical_data = chemical_data.dropna(axis=1)
    chemical_data = chemical_data.drop(
        columns=["Sample location", "Month", "Hydrology", "Hydrology (detailed)"])
    chemical_data["Sample ID"] = chemical_data["Sample ID"].str.replace(
        "20", "")
    chemical_data["Sample ID"] = chemical_data["Sample ID"].str.replace(
        ".", "_")
    chemical_data.columns = chemical_data.columns.str.replace("[", " ")
    chemical_data.columns = chemical_data.columns.str.replace("]", " ")
    chemical_data = chemical_data.set_index("Sample ID")
    chemical_data = chemical_data.sort_values(by="Sample ID")
    chemical_data = chemical_data.replace(",", ".", regex=True)
    chemical_data = chemical_data.apply(pd.to_numeric)
    chemical_data.columns = [column.split(
        " ")[0] for column in chemical_data.columns.values]
    return chemical_data


def prepare_type_data():
    type_data = pd.read_csv(constants.TYPE_DATA_PATH)
    indices_to_names = {}
    for i, name in enumerate(type_data['Phylum (Aggregated)']):
        indices_to_names[i] = name 
    type_data = type_data.dropna(axis=1)
    type_data = type_data.drop(columns=["Combined Abundance"])
    type_data.columns = type_data.columns.str.replace("R ", "R")
    type_data.columns = type_data.columns.str.strip()
    type_data.columns = type_data.columns.str.replace(" ", "_")
    type_data = type_data.reindex(sorted(type_data.columns), axis=1)
    type_data = type_data.apply(pd.to_numeric)
    return type_data, indices_to_names


def prepare_row_data():    
    row_data = pd.read_csv(constants.ROW_DATA_PATH)
    indices_to_names = {}
    for i, name in enumerate(row_data['Order (Aggregated)']):
        indices_to_names[i] = name 
    row_data = row_data.dropna(axis=1)
    row_data = row_data.drop(columns=["Combined Abundance"])
    row_data.columns = row_data.columns.str.replace("R ", "R")
    row_data.columns = row_data.columns.str.strip()
    row_data.columns = row_data.columns.str.replace(" ", "_")
    row_data = row_data.reindex(sorted(row_data.columns), axis=1)
    row_data = row_data.apply(pd.to_numeric)
    return row_data, indices_to_names
    

def prepare_class_data():
    class_data = pd.read_csv(constants.CLASS_DATA_PATH)
    #class_data = class_data[class_data["Combined Abundance"] > 40]
    indices_to_names = {}
    for i, name in enumerate(class_data['Class (Aggregated)']):
        indices_to_names[i] = name 
    class_data = class_data.dropna(axis=1)
    class_data = class_data.drop(columns=["Combined Abundance"])
    class_data.columns = class_data.columns.str.replace("R ", "R")
    class_data.columns = class_data.columns.str.strip()
    class_data.columns = class_data.columns.str.replace(" ", "_")
    class_data = class_data.reindex(sorted(class_data.columns), axis=1)
    class_data = class_data.apply(pd.to_numeric)
    return class_data, indices_to_names

def prepare_genus_data():
    genus_data = pd.read_csv(constants.GENUS_DATA_PATH)
    #genus_data = genus_data[genus_data["Combined Abundance"] > 40]
    indices_to_names = {}
    for i, name in enumerate(genus_data['Genus (Aggregated)']):
        indices_to_names[i] = name 
    genus_data = genus_data.dropna(axis=1)
    genus_data = genus_data.drop(columns=["Combined Abundance"])
    genus_data.columns = genus_data.columns.str.replace("R ", "R")
    genus_data.columns = genus_data.columns.str.strip()
    genus_data.columns = genus_data.columns.str.replace(" ", "_")
    genus_data = genus_data.reindex(sorted(genus_data.columns), axis=1)
    genus_data = genus_data.apply(pd.to_numeric)
    return genus_data, indices_to_names
    
def prepare_species_data():
    species_data = pd.read_csv(constants.SPECIES_DATA_PATH)
    indices_to_names = {}
    for i, name in enumerate(species_data['Species (Aggregated)']):
        indices_to_names[i] = name 
    species_data = species_data.dropna(axis=1)
    species_data = species_data.drop(columns=["Combined Abundance"])
    species_data.columns = species_data.columns.str.replace("R ", "R")
    species_data.columns = species_data.columns.str.strip()
    species_data.columns = species_data.columns.str.replace(" ", "_")
    species_data = species_data.reindex(sorted(species_data.columns), axis=1)
    species_data = species_data.apply(pd.to_numeric)
    return species_data, indices_to_names

def prepare_family_data(drop_unknown_families=False):
    family_data = pd.read_csv(constants.FAMILY_DATA_PATH)
    #print(family_data.columns)
    if drop_unknown_families:
        family_data = family_data[family_data['Family (Aggregated)'].str.contains('Unknown') == False]
    #names = family_data.iloc[:, 0]
    #print(names)
    indices_to_names = {}
    for i, name in enumerate(family_data['Family (Aggregated)']):
        indices_to_names[i] = name 
    family_data = family_data.dropna(axis=1)

    #family_data=family_data.drop(family_data.columns[2], axis=1)
    family_data = family_data.sort_values("Combined Abundance", ascending=False)
    family_data = family_data.drop(columns=["Combined Abundance"])
    family_data.columns = family_data.columns.str.replace("R ", "R")
    family_data.columns = family_data.columns.str.strip()
    family_data.columns = family_data.columns.str.replace(" ", "_")

    family_data = family_data.reindex(sorted(family_data.columns), axis=1)
    family_data = family_data.apply(pd.to_numeric)
    return family_data, indices_to_names