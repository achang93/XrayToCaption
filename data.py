import pandas as pd
import csv
import ast

def load_data(train_data, val_data):
    list_formatted_cols = [
        "image",
        "view",
        "AP",
        "PA",
        "Lateral",
        "text",
        "text_augment"
    ]
    converters = {c: ast.literal_eval for c in list_formatted_cols}
    train_df = pd.read_csv(train_data, converters=converters)
    val_df = pd.read_csv(val_data, converters=converters)

    # remove the first 2 columns
    train_df.drop(train_df.columns[[0, 1]], axis=1, inplace=True)
    val_df.drop(val_df.columns[[0, 1]], axis=1, inplace=True)

    return train_df, val_df

def concat_findings(df):
    df['all_findings'] = df.apply(
        lambda r: ' '.join(r['text'] + r['text_augment']),
        axis=1
    )