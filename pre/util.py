from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

train_file = "./train.csv"
test_file = "./test.csv"
num_cols = ["ps_reg_01", "ps_reg_02", "ps_reg_03","ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]
ignore_cols = ["id", "target", "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04", "ps_calc_05",
               "ps_calc_06", "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11",
               "ps_calc_12", "ps_calc_13", "ps_calc_14","ps_calc_15_bin", "ps_calc_16_bin",
               "ps_calc_17_bin","ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"]


def overview(cfg):
    dfTrain = pd.read_csv(train_file)
    dfTest = pd.read_csv(test_file)
    df = pd.concat([dfTrain, dfTest], sort=False)

    field_size = len(df.columns) - len(ignore_cols)
    feature_dict = {}
    feature_size = 0
    for col in df.columns:
        if col in ignore_cols:
            continue
        elif col in num_cols:
            feature_dict[col] = feature_size
            feature_size += 1
        else:
            unique_val = df[col].unique()
            feature_dict[col] = dict(zip(unique_val, range(feature_size, len(unique_val) + feature_size)))
            feature_size += len(unique_val)

    cfg['field_size'] = field_size
    cfg['feature_size'] = feature_size
    return dfTrain, feature_dict


def process(cfg, split_type=None):
    train_df, feature_dict = overview(cfg)
    label_df = train_df[['target']]
    train_df.drop(['target', 'id'], axis=1, inplace=True)
    feature_idx = train_df.copy()
    feature_val = train_df.copy()
    for col in feature_idx.columns:
        if col in ignore_cols:
            feature_idx.drop(col, axis=1, inplace=True)
            feature_val.drop(col, axis=1, inplace=True)
            continue
        elif col in num_cols:
            feature_idx[col] = feature_dict[col]
        else:
            feature_idx[col] = feature_idx[col].map(feature_dict[col])
            feature_val[col] = 1

    train_idx_df, test_idx_df = train_test_split(feature_idx, test_size=cfg["split"])
    train_val_df, test_val_df = train_test_split(feature_val, test_size=cfg["split"])
    train_label_df, test_label_df = train_test_split(label_df, test_size=cfg["split"])

    train_idx_df, validate_idx_df = train_test_split(train_idx_df, test_size=cfg["split"])
    train_val_df, validate_val_df = train_test_split(train_val_df, test_size=cfg["split"])
    train_label_df, validate_label_df = train_test_split(train_label_df, test_size=cfg["split"])

    train_input = [train_idx_df.values, train_val_df.values]
    train_label = np.array(train_label_df['target'])
    bool_train_labels = train_label != 0

    validate_input = [validate_idx_df.values, validate_val_df.values]
    validate_label = validate_label_df.values

    test_input = [test_idx_df.values, test_val_df.values]
    test_label = test_label_df.values

    return train_input, train_label, bool_train_labels, validate_input, validate_label, test_input, test_label


def oversample(train_input, train_label, bool_train_labels):
    pos_idx = train_input[0][bool_train_labels]
    neg_idx = train_input[0][~bool_train_labels]
    pos_val = train_input[1][bool_train_labels]
    neg_val = train_input[1][~bool_train_labels]
    pos_label = train_label[bool_train_labels]
    neg_label = train_label[~bool_train_labels]

    ids = np.arange(len(pos_idx))
    choices = np.random.choice(ids, len(neg_idx))

    res_pos_idx = pos_idx[choices]
    res_pos_val = pos_val[choices]
    res_pos_label = pos_label[choices]

    resampled_idx = np.concatenate([res_pos_idx, neg_idx], axis=0)
    resampled_val = np.concatenate([res_pos_val, neg_val], axis=0)
    resampled_label = np.concatenate([res_pos_label, neg_label], axis=0)

    order = np.arange(len(resampled_label))
    np.random.shuffle(order)
    return [resampled_idx[order], resampled_val[order]], resampled_label[order]


