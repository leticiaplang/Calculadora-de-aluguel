import pandas as pd


def transform_dict_to_pandas(item_dict, cols):
    df = pd.DataFrame(item_dict, index=[1])
    return df[cols]