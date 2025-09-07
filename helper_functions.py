import pandas as pd  
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import EditedNearestNeighbours

from sdv.metadata import Metadata

class BaseSampler(BaseEstimator):
    """Base class for all custom samplers."""
    
    def __init__(self, random_state: int) -> None:
        self.random_state = random_state
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Fit the sampler to the data and return the resampled data"""
        return self.resample(X, y)
    
    def _get_num_samples(self, y: pd.Series) -> int:
        """Get the number of samples to generate for the minority class for this binary
        classification problem."""
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        return num_samples  

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Resample the data using the specified sampling method"""
        
        # Create copies of the input data
        X_train = X.copy()
        y_train = y.copy()
        
        # Determine the number of samples to generate
        num_samples = self._get_num_samples(y_train)

        # Isolate the minority class and generate synthetic data
        X_minority = X_train[y_train == 1]
        X_upsampled = self._generate_syn_data(X_minority, num_samples)

        # Updating indepent variables and the target variable
        X_sampled = pd.concat([X_train, X_upsampled], axis = 0)
        y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

        return X_sampled, y_sampled
   
    
class NoiseSampler(BaseSampler):
    
    def __init__(self, random_state = None):
        super().__init__(random_state)
           
    def _generate_syn_data(self, X_minority: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        
        sample_results = {}

        for col in X_minority.columns:
            col_mean = X_minority[col].mean()
            col_std = X_minority[col].std()
            
            s = np.random.normal(loc=col_mean, scale = col_std, size = num_samples)
            
            sample_results[col] = s

        X_upsampled = pd.DataFrame(sample_results)
        
        return X_upsampled    


class SDVSampler(BaseSampler):
    """Custom samler for SDV synthesizers."""
    
    def __init__(self, generator, metadata: Metadata, random_state: int) -> None:
        super().__init__(random_state)
        self.generator = generator
        self.metadata = metadata
       
    def _generate_syn_data(self, X_minority: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data using the SDV synthesizer."""
        
        # Creating a copy of the minority class data
        X_resample = X_minority.copy()
        
        # Creating a new instance of the synthesizer within each fold
        synthesizer = self.generator(self.metadata)

        # Fitting the synthesizer to the minority class and generating new observations
        synthesizer.fit(X_resample)
        X_sds = synthesizer.sample(num_samples)
        
        return X_sds
   
    
class SDVENN(SDVSampler):  
    
    def __init__(self, generator, metadata: Metadata, random_state: int = None) -> None:
        super().__init__(generator, metadata, random_state)
        self.enn = EditedNearestNeighbours()

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        
        # Oversample using the original SDV class
        X_sampled, y_sampled = super().resample(X, y)
        
        X_resampled, y_resampled = self.enn.fit_resample(X_sampled, y_sampled)

        return X_resampled, y_resampled
        
               
def display_scores(cv_scores: list, scorings: list) -> pd.DataFrame:
    
    res = {}
   
    for scoring in scorings:
        res[scoring.capitalize()] = [round(score[f"test_{scoring}"].mean(), 3) for score in cv_scores]
    
    res_df = pd.DataFrame(res)
    res_df.index = ['Noise', 'ROS', 'SMOTE', 'CTGAN', 'TVAE', 'SMOTENN', 
                    'CTGANENN', 'CSL']
    
    return res_df
       

