# Import requirment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier # type: ignore

# Load the dataset
df = pd.read_csv('Breast_cancer.csv')

# Get seen in the dataset
print(df.head())
