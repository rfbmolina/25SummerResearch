from sklearn.model_selection import GroupShuffleSplit


# Parameters:
# X, y, groups : pandas objects with identical row order
#     * X      : feature matrix (DataFrame)
#     * y      : target labels  (Series)
#     * groups : group labels   (Series) – rows sharing the same value must stay together in either train or test to prevent data leakage

# Returns

# X_train, X_test, y_train, y_test : train- / test-folds that respect group integrity.

# Notes
# • A single 80 / 20 group-aware split is produced (n_splits = 1).  
# • test_size = 0.20 - ~20 % of groups (not rows) are placed in the test set.  
# • random_state = 42 fixes the shuffling order - fully reproducible. SHOULD PAREMETRISE THIS GROUPSHUFFLESPLIT THE OTHER IS FOR CLASSIFIERS 



def split_data(X, y, groups, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train      = X.iloc[train_idx]
    X_test       = X.iloc[test_idx]
    y_train      = y.iloc[train_idx]
    y_test       = y.iloc[test_idx]
    # groups_train = groups.iloc[train_idx]
    # # Here I make sure that the same group doesn't appear in train and test
    # groups_test  = groups.iloc[test_idx]

    print(f"→ Train  X: {X_train.shape},  y: {y_train.shape}") # groups: {groups_train.shape}")
    print(f"→ Test   X: {X_test.shape},  y: {y_test.shape},") # groups: {groups_test.shape}")
    return X_train, X_test, y_train, y_test #, groups_train, groups_test

