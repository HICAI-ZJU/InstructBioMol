import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from .metabolite_preprocessing import *
from .enzyme_representations import *

import warnings

warnings.filterwarnings("ignore")

import os
from os.path import join


def ESP_predicton(substrate_list, enzyme_list):
    # creating input matrices for all substrates:
    print("Step 1/3: Calculating numerical representations for all metabolites.")
    df_met = metabolite_preprocessing(metabolite_list=substrate_list)

    print("Step 2/3: Calculating numerical representations for all enzymes.")
    enzyme_list = ["" if pd.isnull(value) else value for value in enzyme_list]
    df_enzyme = calcualte_esm1b_ts_vectors(enzyme_list=enzyme_list)

    print("Step 3/3: Making predictions for ESP.")
    # Merging the Metabolite and the enzyme DataFrame:
    df_ES = pd.DataFrame(
        data={"substrate": substrate_list, "enzyme": enzyme_list, "index": list(range(len(substrate_list)))})
    df_ES = merging_metabolite_and_enzyme_df(df_met, df_enzyme, df_ES)
    df_ES_valid, df_ES_invalid = df_ES.loc[df_ES["complete"]], df_ES.loc[~df_ES["complete"]]
    df_ES_valid.reset_index(inplace=True, drop=True)

    # Making predictions
    if len(df_ES_valid) > 0:
        X = calculate_xgb_input_matrix(df=df_ES_valid)
        ESs = predict_ES(X)
        df_ES_valid["Prediction"] = ESs

    df_ES = pd.concat([df_ES_valid, df_ES_invalid], ignore_index=True)
    df_ES = df_ES.sort_values(by=["index"])
    df_ES.drop(columns=["index"], inplace=True)
    df_ES.reset_index(inplace=True, drop=True)
    return (df_ES)


def merging_metabolite_and_enzyme_df(df_met, df_enzyme, df_ES):
    df_ES["GNN FP"], df_ES["enzyme rep"] = "", ""
    df_ES["complete"] = True
    df_ES["metabolite_similarity_score"] = np.nan
    df_ES["metabolite in training set"] = False
    df_ES["#metabolite in training set"] = 0
    for ind in df_ES.index:
        try:
            gnn_rep = list(df_met["substrate_rep"].loc[df_met["metabolite"] == df_ES["substrate"][ind]])[0]
            esm1b_rep = list(df_enzyme["enzyme rep"].loc[df_enzyme["amino acid sequence"] == df_ES["enzyme"][ind]])[0]
        except:
            gnn_rep, esm1b_rep = "", ""

        if gnn_rep == "" or esm1b_rep == "" or df_ES["enzyme"][ind] == "":
            df_ES["complete"][ind] = False
        else:
            df_ES["GNN FP"][ind] = gnn_rep
            df_ES["enzyme rep"][ind] = esm1b_rep
        if gnn_rep != "":
            df_ES["metabolite_similarity_score"][ind] = \
            list(df_met["metabolite_similarity_score"].loc[df_met["metabolite"] == df_ES["substrate"][ind]])[0]
            if df_ES["metabolite_similarity_score"][ind] == 1:
                df_ES["metabolite in training set"][ind] = True
                df_ES["#metabolite in training set"][ind] = \
                list(df_met["#metabolite in training set"].loc[df_met["metabolite"] == df_ES["substrate"][ind]])[0]
    return (df_ES)


def predict_ES(X):
    bst = pickle.load(
        open(join("data/eval_assist/esp_data", "saved_models", "xgboost", "xgboost_model_production_mode_gnn_esm1b_ts.dat"), "rb"))
    feature_names = ["GNN rep_" + str(i) for i in range(100)]
    feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
    dX = xgb.DMatrix(X, feature_names=feature_names)
    ESs = bst.predict(dX)
    return (ESs)


def calculate_xgb_input_matrix(df):
    ESM1b = np.array(list(df["enzyme rep"]))
    fingerprints = ();

    for ind in df.index:
        ecfp = np.array(df["GNN FP"][ind])
        fingerprints = fingerprints + (ecfp,);
    fingerprints = np.array(fingerprints)

    print(fingerprints.shape, ESM1b.shape)
    X = np.concatenate([fingerprints, ESM1b], axis=1)
    return (X)
