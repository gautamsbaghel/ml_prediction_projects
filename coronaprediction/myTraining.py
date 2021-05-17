import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


if __name__=="__main__":
    #read data
    df=pd.read_csv('corona_dataset.csv')

    #split dataframe
    X = df.iloc[:,:5].values  #values is used to convert it into array
    Y =df.iloc[:,5].values
    #splitting x and y in training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    #calling logisticregression
    clf= LogisticRegression()
    #fitting training data to algorithm
    clf.fit(X_train,Y_train)

    #pickle is used to save the model (saving model data)
    #open a file ,whereyou want to store data
    file=open('model.pkl', 'wb')

    #dump information to that file
    pickle.dump(clf, file)
    file.close()
    
