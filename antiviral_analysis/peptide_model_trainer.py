import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


class PeptideTrainer:
    def __init__(self, df, do_even_sampling=True, upsample_method="smote"):
        self.data = df.fillna(0)
        self.num_features = len(self.data.columns)
        self.do_even_sampling = do_even_sampling
        self.upsample_method = upsample_method
        self.train_test_set = [None, None, None, None]
        self.model = None
        self.model_type = None
        self.model_losses = None
        
        
    def prep_train_test_data(self, inplace=True):
        scaler = MinMaxScaler()
        X = np.array(self.data.drop("AV label",axis=1).values)
        y = np.array(self.data["AV label"].values)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if inplace:
            self.train_test_set = [X_train, X_test, y_train, y_test]
        return [X_train, X_test, y_train, y_test]
        
        
    def even_sampling(self, inplace=True):
        [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.prep_train_test_data()
        if self.upsample_method=="smote":
            sm = SMOTE()
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
            train_test_set_even = [X_train_sm, X_test, y_train_sm, y_test]
        
        elif self.upsample_method=="undersampling":
            rus = RandomUnderSampler(random_state=0)
            X_train_rus, y_train_rus = rus.fit_resample(X_train,y_train)
            train_test_set_even = [X_train_rus, X_test, y_train_rus, y_test]
            
        if inplace:
            self.train_test_set = train_test_set_even
        return train_test_set_even
    
    
    def train_nn_model(self, inplace=True, save_model=True):
        if self.do_even_sampling:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.even_sampling()
        else:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.prep_train_test_data()
        nn = Sequential()
        nn.add(Dense(self.num_features,activation="relu"))
        nn.add(Dropout(0.4))
        nn.add(Dense(int(self.num_features/1.66),activation="relu"))
        nn.add(Dropout(0.3))
        nn.add(Dense(int(self.num_features/3.50),activation="relu"))
        nn.add(Dropout(0.2))
        if self.num_features >= 20:
            nn.add(Dense(int(self.num_features/8.33),activation="relu"))
            nn.add(Dropout(0.1))
        nn.add(Dense(1,activation="sigmoid"))
        nn.compile(loss="binary_crossentropy",optimizer="adam")
        
        early_stop = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=35)
        
        nn.fit(x=X_train,y=y_train,epochs=300,validation_data=(X_test,y_test),callbacks=[early_stop])
        
        if inplace:
            self.model = nn
            self.model_type = "nn"
            self.model_losses = pd.DataFrame(nn.history.history)
        if save_model:
            time_now = str(datetime.now())
            print(f"Saving model with filename '{self.upsample_method}_nn_model_AV_{time_now}.h5'")
            nn.save(f"{self.upsample_method}_nn_model_AV_{time_now}.h5")
        return nn
    
    def train_rfc_model(self, inplace=True, save_model=True):
        if self.do_even_sampling:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.even_sampling()
        else:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.prep_train_test_data()
        param_grid = {
            'bootstrap': [True], 'max_depth': [70, 80, 90, 100, 110, 120],
            'max_features': [2, 3, 4],'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12], 'n_estimators': [100, 200, 300, 500, 1000]
            }
        # Create an rfc based model
        rfc_estimator = RandomForestClassifier()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rfc_estimator, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 1)
        grid_search.fit(X_train, y_train)
        rfc = grid_search.best_estimator_
        rfc.fit(X_train, y_train)
        if inplace:
            self.model = rfc
            self.model_type = "rfc"
        if save_model:
            time_now = str(datetime.now()) 
            print(f"Saving model with filename '{self.upsample_method}_rfc_model_AV_{time_now}.joblib'")
            dump(rfc, f"{self.upsample_method}_rfc_model_AV_{time_now}.joblib")
        return rfc

    def train_svm_model(self, inplace=True, save_model=True):
        if self.do_even_sampling:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.even_sampling()
        else:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.prep_train_test_data()
        param_grid = {'C': [0.1,1, 10, 50, 100], 'gamma': [1,0.1, 0.05,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
        # Create an svc based model
        svc_estimator = SVC()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = svc_estimator, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 1)
        grid_search.fit(X_train, y_train)
        svc = grid_search.best_estimator_
        svc.fit(X_train, y_train)
        if inplace:
            self.model = svc
            self.model_type = "svc"
        if save_model:
            time_now = str(datetime.now()) 
            print(f"Saving model with filename '{self.upsample_method}_svc_model_AV_{time_now}.joblib'")
            dump(svc, f"{self.upsample_method}_svc_model_AV_{time_now}.joblib")
        return svc
            
    def evaluate_model(self, show_plots=True):
        if self.do_even_sampling:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.even_sampling()
        else:
            [X_train, X_test, y_train, y_test] = self.train_test_set if \
                                                all([val is not None for val in self.train_test_set]) \
                                                    else self.prep_train_test_data()
                    
        model = self.train_model() if self.model is None else self.model
        if self.model_type == "nn":
            pred = np.where(model.predict(X_test) > 0.5, 1, 0)
        elif self.model_type == "rfc" or self.model_type == "svc":
            pred = model.predict(X_test)
        
        if self.upsample_method == "smote":
            print("Evaluation reports for SMOTE upsampling\n")
        elif self.upsample_method == "undersampling":
            print("Evaluation reports for random undersampling")
        print(confusion_matrix(y_test, pred))
        print("\n")
        print(classification_report(y_test, pred))
        print("\n")
        
        if show_plots and self.model_type == "nn":
            self.model_losses.plot()
