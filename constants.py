from pathlib import Path

SEED = 42

MAIN_DATA_PATH = Path("data")

CHEMICAL_DATA_PATH = MAIN_DATA_PATH / \
    "Dane chemiczne 2017-2018 Rev_do sta_halfLOD_ETI.csv"

TYPE_DATA_PATH = MAIN_DATA_PATH / "merged_all_POP_ETI_typ.csv"

ROW_DATA_PATH = MAIN_DATA_PATH / "merged_all_POP_ETI_rzad.csv"

CLASS_DATA_PATH = MAIN_DATA_PATH / "merged_all_POP_ETI_klasa.csv"


GENRA_DATA_PATH = MAIN_DATA_PATH / "merged_all_POP_ETI_rodzaj.csv"