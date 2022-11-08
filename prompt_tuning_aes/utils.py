import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#fix randomness
np.random.seed(42)


score_range = {
    1:(2,12),
    2:(1,6),
    3:(0,3),
    4:(0,3),
    5:(0,4),
    6:(0,4),
    7:(0,30),
    8:(0,60)
}

def scale_score(df):
    dataframes = []
    for prompt_id, group in df.groupby('essay_set'):
        min_score = score_range[prompt_id][0]
        max_score = score_range[prompt_id][1]

        scaled_group = group.copy(deep=True)
        scaled_group['domain1_score'] = ((group['domain1_score'] - min_score) / (max_score - min_score))
        dataframes.append(scaled_group)
    sclaed_df = pd.concat(dataframes, axis=0)

    return sclaed_df

def kfold_split(target_df, n_splits = 5):
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    splits = [(train_index, test_index) for train_index, test_index in kfold.split(target_df)]
    target_train_df = target_df.iloc[splits[0][0]].copy(deep=True)
    test_df = target_df.iloc[splits[0][1]].copy(deep=True)

    return target_train_df, test_df

def read_data(dataset_path, source_id, target_id, target_n = 0):
    #dataset_path: excel file path .xlsx
    df = pd.read_excel(dataset_path)
    df = df[["essay_id","essay_set","essay","domain1_score"]]
    # scale the score for each prompt
    scaled_df = scale_score(df)

    # select data for source prompt id & target prompt id
    source_df = scaled_df[scaled_df["essay_set"]==source_id].copy(deep=True)
    target_df = scaled_df[scaled_df["essay_set"]==target_id].copy(deep=True)

    # split target prompt (essay, score) pair into k folds
    target_train_df, test_df = kfold_split(target_df)

    # construct training data: source prompt (essay, score) pairs + target prompt sampled essays (with configurable random seed)
    sampled_target = target_train_df.sample(n=target_n, replace=False, random_state=42)
    train_df = pd.concat([source_df, sampled_target], ignore_index=True)

    #split training dataset into train, valid set
    train_df, valid_df = train_test_split(train_df, test_size=0.1, shuffle=True, random_state=42)

    return train_df, valid_df, test_df


if __name__ == "__main__":

    cross_settings = [(1,2),(3,4),(5,6),(7,8)]
    target_ns = [0,10,25,50,100]
    for source_id, target_id in cross_settings:
        for target_n in target_ns:
            train_df, valid_df, test_df= read_data("asap_dataset/training_set_rel3.xlsx", source_id, target_id, target_n=target_n)
            train_df.to_excel("asap_dataset/processed_dataset/source_{}_target_{}/train_with_{}_targets.xlsx".format(source_id, target_id, target_n))
            valid_df.to_excel("asap_dataset/processed_dataset/source_{}_target_{}/valid_with_{}_targets.xlsx".format(source_id, target_id, target_n))
        test_df.to_excel("asap_dataset/processed_dataset/source_{}_target_{}/test.xlsx".format(source_id, target_id))


