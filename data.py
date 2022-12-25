import constants

import pandas as pd


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
    return chemical_data


def prepare_type_data():
    type_data = pd.read_csv(constants.TYPE_DATA_PATH)
    type_data = type_data.dropna(axis=1)
    type_data = type_data.drop(columns=["Combined Abundance"])
    type_data.columns = type_data.columns.str.replace("R ", "R")
    type_data.columns = type_data.columns.str.strip()
    type_data.columns = type_data.columns.str.replace(" ", "_")
    type_data = type_data.reindex(sorted(type_data.columns), axis=1)
    type_data = type_data.apply(pd.to_numeric)
    return type_data
