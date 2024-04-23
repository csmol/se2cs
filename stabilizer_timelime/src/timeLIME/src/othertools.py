import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from scipy import stats
import time
import itertools
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder## This file contains all tools needed to run planners.
import warnings

warnings.filterwarnings("ignore")

def readfile(fname):
    file = pd.read_csv(fname, sep=',')
    file.drop(columns=file.columns[0], inplace=True)
    result = []
    N = file.shape[0]
    for i in range(N):
        temp = []
        for k in range(file.shape[1]):
            if pd.isnull(file.iloc[i, k]):
                continue
            else:
                temp.append(file.iloc[i, k])
        result.append(temp)
    return result

def list2dataframe(lst):
    return pd.DataFrame(lst)


def prepareData(fname):
    file = os.path.join("data", fname)
    df = pd.read_csv(file,sep=',')
    cols=list(df.columns)
    cols = ['dates', 'monthly_stargazer', 'monthly_commits', 'monthly_closed_issues', 'monthly_closed_PRs', 'monthly_contributors', 'monthly_open_issues', 'monthly_open_PRs']


    df = pd.DataFrame(df[cols])
    return df

def bugs(fname):
    file = os.path.join("data", fname)
    df = pd.read_csv(file,sep=',')
    return df.iloc[:,3]

def get_index(name):
    feature = ['monthly_stargazer', 'monthly_commits', 'monthly_closed_issues', 'monthly_closed_PRs', 'monthly_contributors', 'monthly_open_issues']
    for i in range(len(feature)):
        if name ==feature[i]:
            return i
    return -1

def translate1(sentence,name):
    # do not aim to change the column
    lst = sentence.strip().split(name)
    left,right= 0,0
    if lst[0] == '':
        del lst[0]
    if len(lst)==2:
        if '<=' in lst[1]:
            aa=lst[1].strip(' <=')
            right= float(aa)
        elif '<' in lst[1]:
            aa=lst[1].strip(' <')
            right = float(aa)
        if '<=' in lst[0]:
            aa=lst[0].strip(' <=')
            left = float(aa)
        elif '<' in lst[0]:
            aa=lst[0].strip(' <')
            left = float(aa)
    else:
        if '<=' in lst[0]:
            aa=lst[0].strip(' <=')
            right = float(aa)
            left = 0
        elif '<' in lst[0]:
            aa=lst[0].strip(' <')
            right = float(aa)
            left = 0
        if '>=' in lst[0]:
            aa=lst[0].strip(' >=')
            left = float(aa)
            right = 1
        elif '>' in lst[0]:
            aa=lst[0].strip(' >')
            left = float(aa)
            right = 1
    return left, right


def flip(data_row,local_exp,ind,clf,cols,n_feature = 5,actionable = None):
    counter = 0
    rejected = 0
    cache = []
    trans = []
    # Store feature index in cache.
    cnt, cntp,cntn = [],[],[]
    for i in range(0,len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        if ind[i][1]>0.01:
            cntp.append(i)
            cnt.append(i)
        elif ind[i][1]<-0.01:
            cntn.append(i)
            cnt.append(i)

    record =[0 for n in range(6)]
    tem = data_row.copy()
    result =  [[ 0 for m in range(2)] for n in range(6)]

    for j in range(0,len(local_exp)):
        act = True
        index = get_index(cols[cache[j][0]])
        if actionable:
            if actionable[index]==0:
                act=False
        l,r = translate1(trans[j][0],cols[cache[j][0]])
        if j in cnt and counter<n_feature and act:
            if j in cntp:
                result[cache[j][0]][0],result[cache[j][0]][1] = 0,tem[index]
                record[index]=-1
            else:
                result[cache[j][0]][0],result[cache[j][0]][1] = tem[index],1
                record[index]=1
            counter+=1
        else:
            if act:
                result[cache[j][0]][0],result[cache[j][0]][1] = tem[index]-0.005,tem[index]+0.005
                # result[cache[j][0]][0],result[cache[j][0]][1] = tem[index],tem[index]
            else:
                result[cache[j][0]][0],result[cache[j][0]][1] = tem[index]-0.05,tem[index]+0.05
                # result[cache[j][0]][0],result[cache[j][0]][1] = tem[index],tem[index]
    return tem,result,record

def hedge(arr1,arr2):
    # returns a value, larger means more changes
    s1,s2 = np.std(arr1),np.std(arr2)
    m1,m2 = np.mean(arr1),np.mean(arr2)
    n1,n2 = len(arr1),len(arr2)
    num = (n1-1)*s1**2 + (n2-1)*s2**2
    denom = n1+n2-1-1
    sp = (num/denom)**.5
    delta = np.abs(m1-m2)/sp
    c = 1-3/(4*(denom)-1)
    return delta*c


def norm (df1,df2):
    # min-max scale the dataset
    X1 = df1.iloc[:,:-1].values
    mm = MinMaxScaler()
    mm.fit(X1)
    X2 = df2.iloc[:,:-1].values
    X2 = mm.transform(X2)
    df2 = df2.copy()
    df2.iloc[:,:-1] = X2

    return df2


def overlap(plan,actual): # Jaccard similarity function
    cnt = 6
    right = 0
    for i in range(0,len(plan)):
        if isinstance(plan[i], float):
            if np.round(actual[i],4)== np.round(plan[i],4):
                right+=1
        else:
            if actual[i]>=0 and actual[i]<=1:
                if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
                    right+=1
            elif actual[i]>1:
                if plan[i][1]>=1:
                    right+=1
            else:
                if plan[i][0]<=0:
                    right+=1
    return right/cnt


def similar(ins):
    out = []
    for i in range(0,len(ins.as_list(label = 1))):
        out.append(ins.as_list(label = 1)[i][0])
    return out


def overlap1(ori,plan,actual): # Jaccard similarity function
    cnt = 6
    right = 0
    for i in range(0,len(plan)):
        if isinstance(plan[i], list):
            if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
                right+=1
        else:
            if actual[i]==plan[i]:
                right+=1
    return right/cnt


def size_interval(plan):
    out=[]
    for i in range(len(plan)):
        if not isinstance(plan[i],float):
            out.append(plan[i][1]-plan[i][0])
        else:
            out.append(0)
    return out


def track1(old,new):
    rec=[]
    for i in range(len(old)):
        if old[i][0]<=new[i]<=old[i][1]:
            rec.append(0)
        elif old[i][0]>new[i]:
            rec.append(-1)
        else:
            rec.append(1)
    return rec

def track(old, new):
    rec = []
    for i in range(len(old)):
        if old[i] != new[i]:
            if new[i] > old[i]:
                rec.append(1)
            else:
                rec.append(-1)
        else:
            rec.append(0)
    return rec

def frequentSet(name):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df11, df22)
    df3n = norm(df11, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]
    records = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if df3.iloc[j, 0] == df2.iloc[i, 0]:
                actual = X_test2.values[j]
                old = X_test1.values[i]
                rec = track(old, actual)
                records.append(rec)
                break

    for i in range(0, len(y_train1)):
        for j in range(0, len(y_test1)):
            if df2.iloc[j, 0] == df1.iloc[i, 0]:
                actual = X_test1.values[j]
                old = X_train1.values[i]
                rec = track(old, actual)
                records.append(rec)
                break
    return records


def transform(df2,lo,col):
    # transform single column of index 'col' by discretized data
    df22 = df2.copy()
    low = lo[col]
    start = low[0]
    for i in range(df22.shape[0]):
        for j in range(len(low)):
            if df22.iloc[i,col]>=low[j]:
                start = low[j]
            if df22.iloc[i,col]<=low[j]:
                end = low[j]
                df22.iloc[i,col] = (start+end)/2
                break
    return df22


def abcd(ori,plan,actual,act):
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(ori)):
        if act[i]!=0:
            if isinstance(plan[i], float):
                if plan[i]==actual[i]:
                    tp+=1
                else:
                    fp+=1
            else:
                if plan[i][0]<=actual[i]<=plan[i][1]:
                    tp+=1
                else:
                    fp+=1
        else:
            if isinstance(plan[i], float):
                if plan[i]==actual[i]:
                    tn+=1
                else:
                    fn+=1
            else:
                if plan[i][0]<=actual[i]<=plan[i][1]:
                    tn+=1
                else:
                    fn+=1
    return tp,tn,fp,fn


def convert_to_itemset(df):
    itemsets=[]
    for i in range(df.shape[0]):
        item=[]
        temp = df.iloc[i,:]
        for j in range(df.shape[1]):
            if temp[j]==1:
                item.append("inc"+str(j))
            elif temp[j]==-1:
                item.append("dec"+str(j))
        if len(item)>0:
            itemsets.append(item)
    return itemsets


def mine_rules(itemsets):
    test = itemsets.copy()
    te = TransactionEncoder()
    te_ary = te.fit(test).transform(test,sparse=True)
    df =  pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    rules = apriori(df, min_support=0.001, use_colnames=True)
    return rules


def get_support(string, rules):
    for i in range(rules.shape[0]):
        if set(rules.iloc[i,1]) == set(string):
            return rules.iloc[i,0]
    return 0


def find_supported_plan(plan,rules,top=5):
    proposed = []
    max_change=top
    max_sup=0
    result_id=[]
    pool=[]
    for j in range(len(plan)):
        if plan[j]==1:
            result_id.append(j)
            proposed.append("inc"+str(j))
        elif plan[j]==-1:
            result_id.append(-j)
            proposed.append("dec"+str(j))
    while (max_sup==0):
        pool = list(itertools.combinations(result_id, max_change))
        for each in pool:
            temp = []
            for k in range(len(each)):
                if each[k]>0:
                    temp.append("inc"+str(each[k]))
                elif each[k]<0:
                    temp.append("dec"+str(-each[k]))
            # print('temp',temp)
            temp_sup = get_support(temp,rules)
            if temp_sup>max_sup:
                max_sup = temp_sup
                result_id = each
        max_change-=1
        if max_change<=0:
            print("Failed!!!")
            break
    return result_id
