from othertools import *
import pandas as pd
from sklearn.svm import SVR
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler
import warnings
import csv
from datetime import datetime

warnings.filterwarnings("ignore")

def TL(name,par,target, rules,smote=False,act=False):

    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 6
    deltas = []
    for j in range(0, len(files) - 2):
        df1, cols = prepareData(files[j],target)
        df2, cols = prepareData(files[j + 1],target)
        for i in range(1, 7):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))

    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    actionable = []

    for i in range(0, len(deltas)):
        if i in deltas[0:4]:
            actionable.append(1)
        else:
            actionable.append(0)
    print("actionable")
    print(actionable)
    df1, cols = prepareData(name[0],target)
    df2, cols = prepareData(name[1],target)
    df3, cols = prepareData(name[2],target)

    bug1 = bugs(name[0],target)
    bug2 = bugs(name[1],target)
    bug3 = bugs(name[2],target)

    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    seen = []
    seen_id = []
    par = 6
    clf1 = RandomForestRegressor()

    if smote:
        print("Smote")
        sm = SMOTE()
        X_train1_s, y_train1_s = X_train1, y_train1
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=pd.DataFrame(X_train1_s).values, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression')
    else:
        print("No smote")
        clf1.fit(X_train1, y_train1)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression')

    file_path_1 = "se2cs/stabilizer_timelime/src/timeLIME/resuts/output.csv"
    # Writing data to CSV
    with open(file_path_1, mode='a', newline='') as file:
        writer = csv.writer(file)
        current_date = datetime.now().date()
        writer.writerow(['Date'] + [current_date])
        writer.writerow(['name'] + [name[0]])
         # Feature names
        feature_names = cols[1: 6]
        # Write header row
        writer.writerow(['Month'] + feature_names)

        for i in range(0, len(y_test1)):
            for j in range(0, len(y_test2)):
                actual = X_test2.values[j]
                if True:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict,
                                                        num_features=6,
                                                        num_samples=5000)
                    ind = ins.local_exp[1]

                    temp = X_test1.values[i].copy()

                    if act:
                        if target == "monthly_open_issues" or target == "monthly_open_PRs":
                            tem, plan, rec = flip_dec(target, temp, ins.as_list(), ind, clf1, df1n.columns, par, actionable=actionable)
                        else:
                            tem, plan, rec = flip_inc(target, temp, ins.as_list(), ind, clf1, df1n.columns, par, actionable=actionable)

                    else:
                        if target == "monthly_open_issues" or target == "monthly_open_PRs":
                            tem, plan, rec = flip_dec(target, temp, ins.as_list(), ind, clf1, df1n.columns, par, actionable=None)
                        else:
                            tem, plan, rec = flip_inc(target, temp, ins.as_list(), ind, clf1, df1n.columns, par, actionable=None)


                    for k in range(len(plan)):
                        if plan[k] == [0, 0] or plan[k] == [0, 0.0]:
                            plan[k] = [-0.05, 0.05]

                    # Write data
                    writer.writerow([f'Month_{i+1}'] + rec)

                    print("rec:", rec)
                    print("==============================================================")
                    score.append(overlap(plan, actual))
                    size.append(size_interval(plan))
                    score2.append(overlap(plan, actual))
                    records.append(rec)
                    tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                    print("")
                    matrix.append([tp, tn, fp, fn])

                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    return score, bugchange, size, score2, records, matrix


def historical_logs(name, par, target, explainer=None, smote=False, small=0.05, act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 6
    deltas = []
    for j in range(0, len(files) - 2):
        df1,cols = prepareData(files[j],target)
        df2, cols = prepareData(files[j + 1],target)
        for i in range(1, 7):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:4]:
            actionable.append(1)
        else:
            actionable.append(0)

    df1, cols = prepareData(name[0],target)
    df2, cols = prepareData(name[1],target)
    df3, cols = prepareData(name[2],target)
    bug1 = bugs(name[0],target)
    bug2 = bugs(name[1],target)
    bug3 = bugs(name[2],target)
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    old_change = []
    new_change = []
    par = 0
    clf1 = RandomForestRegressor()

    if smote:
        sm = SMOTE()

        X_train1_s, y_train1_s = X_train1, y_train1
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=pd.DataFrame(X_train1_s).values, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression', sample_around_instance=True)
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression', sample_around_instance=True)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if True:
                actual = X_test2.values[j]
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict,
                                                 num_features=6,
                                                 num_samples=5000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                if act:
                    if target == "monthly_open_issues" or target == "monthly_open_PRs":
                        tem, plan, rec = flip_dec(target, temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=actionable)
                    else:
                        tem, plan, rec = flip_inc(target, temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=actionable)
                else:
                    if target == "monthly_open_issues" or target == "monthly_open_PRs":
                        tem, plan, rec = flip_dec(target, temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=None)
                    else:
                        tem, plan, rec = flip_inc(target, temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=None)
                o = track1(plan[:-1], temp)
                n = track1(plan[:-1], actual)
                old_change.append(o)
                new_change.append(n)

                break

    return old_change, new_change
