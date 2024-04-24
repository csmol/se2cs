from planner import *
import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")

def main():
    paras = [True]
    explainer = None
    print("RUNNING FREQUENT ITEMSET LEARNING")

    # Path to your data folder
    data_folder = 'se2cs/stabilizer_timelime/src/timeLIME/data'
    # List to store the file names
    fnames = []
    # List all files in the data folder
    files = os.listdir(data_folder)
    # Split the list of files into chunks of 3
    split_files = [files[i:i+3] for i in range(0, len(files), 3)]
    # Append the split files to fnames
    for split in split_files:
        fnames.append(split)

    # Path to your CSV file
    target_file = 'se2cs/stabilizer_timelime/src/timeLIME/target/target.csv'

    # Read the CSV file
    df = pd.read_csv(target_file)

    # Extract the value of the 'Target' column
    target = df['Target'].values[0]

    old, new = [], []
    for par in paras:
        for name in fnames:
            o, n = historical_logs(name, 6, target, explainer, smote=True, small=.03, act=par)
            old.append(o)
            new.append(n)
    everything = []
    for i in range(len(new)):
        everything.append(old[i] + new[i])

    # TimeLIME planner
    paras = [True]
    explainer = None
    scores_t, bcs_t = [], []
    size_t, score_2t = [], []
    records2 = []
    con_matrix1 = []
    i = 0
    print("-----------------")
    for par in paras:
        for name in fnames:
            df = pd.DataFrame(everything[i])
            i += 1
            itemsets = convert_to_itemset(df)
            te = TransactionEncoder()
            te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
            df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
            rules = apriori(df, min_support=0.001, max_len=5, use_colnames=True)

            score, bc, size, score_2, rec, mat = TL(name, 6, target, rules, smote=True, act=par)
            scores_t.append(score)
            bcs_t.append(bc)
            size_t.append(size)
            score_2t.append(score_2)
            records2.append(rec)
            con_matrix1.append(mat)

    pd.DataFrame(score_2t).to_csv("rq1_TimeLIME.csv")
    pd.DataFrame(scores_t).to_csv("rq2_TimeLIME.csv")
    pd.DataFrame(bcs_t).to_csv("rq3_TimeLIME.csv")

    return

if __name__ == "__main__":
    main()

