import pandas as pd  
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

class NoiseSampler(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit_resample(self, X, y):
        return self.resample(X, y)
    
    def _get_num_samples(self, y):
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    def _generate_syn_data(self, X_minority, num_samples):
        
        sample_results = {}

        for col in X_minority.columns:
            col_mean = X_minority[col].mean()
            col_std = X_minority[col].std()
            
            s = np.random.normal(loc = col_mean, scale = col_std, size = num_samples)
            
            sample_results[col] = s

        df_sampled = pd.DataFrame(sample_results)
        
        return df_sampled
    
    def resample(self, X, y):
        """Function to generate synthetic samples for the minority class using noise"""
        
        X_train = X.copy()
        y_train = y.copy()
        
        num_samples = self._get_num_samples(y_train)

        # Concatting X and y to ensure we oversample the correct observations
        train = pd.concat([X_train, y_train], axis = 1)
        train = train[train['Class'] == 1]
        
        # Generate synthetic samples
        df_sampled = self._generate_syn_data(train, num_samples)
        X_sampled = pd.concat([X_train, df_sampled], axis = 0)

        # Also updating the target variable
        y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

        return X_sampled, y_sampled


def get_scores(y_test, y_preds, sampling_procedures):
    """Calculates precison, recall, F1, and ROC AUC scores for each sampling procedure.
    """
    scores_dict = {}
    
    for i in range(len(y_preds)):
        precision = precision_score(y_test, y_preds[i], average = 'macro')
        recall = recall_score(y_test, y_preds[i], average = 'macro')
        f1 = f1_score(y_test, y_preds[i], average = 'macro')
        roc_auc = roc_auc_score(y_test, y_preds[i])
        
        scores_dict[sampling_procedures[i]] = [precision, recall, f1, roc_auc]
    
    scores_df = pd.DataFrame(scores_dict).T
    scores_df.columns = ['Precision', 'Recall', 'F1', 'ROC_AUC']
    
    return scores_df