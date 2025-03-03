import pandas as pd  
import numpy as np

def noise_sampler(X_train, y_train):
    """Function to generate synthetic samples for the minority class using noise"""
    
    # Determine the number of samples to generate
    y_values = y_train.value_counts()
    num_samples = y_values[0] - y_values[1]

    # Concatting X and y to ensure we oversample the correct observations
    train = pd.concat([X_train, y_train], axis = 1)
    train = train[train['Class'] == 1]

    sample_results = {}

    for col in X_train.columns:
        col_mean = train[col].mean()
        col_std = train[col].std()
        
        s = np.random.normal(loc = col_mean, scale = col_std, size = num_samples)
        
        sample_results[col] = s

    # Concatting the final dataframe to form the new training set
    df_sampled = pd.DataFrame(sample_results)
    X_sampled = pd.concat([X_train, df_sampled], axis = 0)

    # Also updating the target variable
    y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

    return X_sampled, y_sampled
