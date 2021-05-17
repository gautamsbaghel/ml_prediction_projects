import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

if __name__=="__main__":
    #read data
    emp_sal = pd.read_csv('salary_data.csv')

    #assign independent and dependent variable
    x =emp_sal.iloc[:,:-1].values
    y = emp_sal.iloc[:,-1].values

    #splitting dataset into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=41)

    #calling linearregression
    lr = LinearRegression()
    #fitting training data into algorithm
    lr.fit(x_train,y_train)

    #pickle is used to save the model (saving model data)
    #open a file ,whereyou want to store data
    file=open('salary.pkl', 'wb')

    #dump information to that file
    pickle.dump(lr, file)
    file.close()