

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preprocessing(input: str, inputVars: int, outputVars: int):
    """Preprocessing a csv

    Parameters
    ----------
    input : string that points to the csv
    inputVars: int that defines number of input variables
    outputVars: int that defines number of output variables
    """

    # Importing the dataset
    dataset = pd.read_csv(input)
    X = dataset.iloc[:, :-outputVars].values
    y = dataset.iloc[:, inputVars].values

    # Taking care of missing data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer = imputer.fit(X[:, 1:inputVars])
    X[:, 1:inputVars] = imputer.transform(X[:, 1:inputVars])
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    """from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)"""

    return X_train, X_test, y_train, y_test 