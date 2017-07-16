# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:03:55 2017

@author: Sreerag
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from datetime import date 
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler
from sklearn.tree import export_graphviz
# extraction
##***ENTER DATASET PATH HERE***##
Vessel_P = pd.read_excel('C:/VESSEL PRODUCTIVITY.xlsx')
VQ = pd.read_excel('C:/VESSEL SEQUENCE.xlsx')
av = pd.read_csv('C:/AVAILABILITY.csv',low_memory= False)
df = pd.read_csv('C:/ATTRIBUTES.csv', low_memory = False)
df.isnull().sum()
##########################################################################################
#Vessel Sequence + Vessel Productivity
##########################################################################################
dfV= Vessel_P.groupby('Vessel Name').Volume.sum()
Vessel_P.rename(columns={'Standby Hours': 'Standby_Hours'}, inplace=True)
dfH= Vessel_P.groupby('Vessel Name').Standby_Hours.sum()
dfR= Vessel_P.groupby('Vessel Name').Restows.sum()
result = pd.concat([dfV, dfH,dfR], axis = 1, join='inner')

VQ.rename(columns={'Vessel Code': 'Vessel_Code'}, inplace=True)
VQ.drop('Vessel Call Year',axis=1,inplace = True)
VQ.drop('Vessel Call Sequence',axis=1,inplace = True)
VQ.drop('Voyage Number',axis=1,inplace = True)
VQ.drop_duplicates(inplace = True)
VQ.set_index('Vessel Name',inplace =True)

DS = pd.concat([result,VQ],axis =1,join='inner')
###########################################################################################

av.head(3)
av.rename(columns={'Update Time': 'Update_Time'}, inplace=True)
avF= av.groupby('Container Number').Update_Time.max()
avF = avF.to_frame()
#df = pd.read_csv('C:/rawdata.csv', low_memory = False)
df.drop_duplicates(inplace = True)
avF.drop_duplicates(inplace = True)
df.head(3)
avF.head(3)

df.set_index('Container Number',inplace=True)
Fds = df.join(avF)
Fds.head(3)
Fds.reset_index(inplace=True)
Fds.set_index('Vessel Code',inplace=True)
DS.reset_index(inplace=True)
DS.rename(columns={'Vessel_Code': 'Vessel Code'}, inplace=True)
DS.set_index('Vessel Code',inplace=True)
###########################################################################################
#final functional data set
###########################################################################################
#Z = pd.concat([Fds,DS],axis =1,join='inner')

Z = Fds.join(DS)
Z.head(1000)
Z.index
Z.reset_index(inplace=True)
Z.dropna(inplace=True)

#len(Z)
Z.head(3)
#Z.to_csv('C:/project/out.csv', sep=',')

y1 =Z.loc[0:,'Update_Time'].values  # read gang sequence
y2 =Z.loc[0:,'In Date'].values  # read indate time

y = []
for i in range(len(y1)):
    p= list(('{0:.13f}'.format(y1[i])).strip())
    #p= list(str(y1[0]).strip())
    p= p[0:14]
    yearp =int(''.join(p[0:4]))
    mp = int(''.join(p[4:6]))
    dtp = int(''.join(p[6:8]))
    t1 = date(yearp,mp,dtp)
    
    q= list(('{0:.13f}'.format(y2[i])).strip())
    #q= list(y2[i].strip())
    q=(q[0:14])
    yearq =int(''.join(q[0:4]))
    mq = int(''.join(q[4:6]))
    dtq = int(''.join(q[6:8]))
   
    t2 = date(yearq,mq,dtq)
    #T2.append(t2)
    targetdays = abs((t1-t2).days)
    y.append(targetdays)
   
y = np.array(y)
df = Z.loc[0:,['Vessel Bay','Vessel Row','Vessel Tier','Yard Location Updates',
               'Volume','Standby_Hours','Restows']]


#Data preprocessing

df['Total']=y
df.isnull().sum()
df['Total'].replace(' ', np.nan, inplace=True)
df['Vessel Bay'].replace(' ', np.nan, inplace=True)
df['Vessel Row'].replace(' ', np.nan, inplace=True)
df['Vessel Tier'].replace(' ', np.nan, inplace=True)
df['Yard Location Updates'].replace(' ', np.nan, inplace=True)
df['Volume'].replace(' ', np.nan, inplace=True)
df['Standby_Hours'].replace(' ', np.nan, inplace=True)
df['Restows'].replace(' ', np.nan, inplace=True)

df= df.drop_duplicates()
df = df.dropna()
#boxplot with outlier
plt.boxplot(y)
# don't show outlier points
plt.figure()
plt.boxplot(y, 0, '')


df = df.loc[df['Total']<15]

#df[2] = df[2].astype('category')
#df[1] = df[1].astype('category')
#df[0] = df[0].astype('category')
y= df['Total']
X= df.loc[0:,['Vessel Bay','Vessel Row','Vessel Tier','Yard Location Updates',
               'Volume','Standby_Hours','Restows']]
scaler = StandardScaler()
scaler.fit(X) 
X_std = scaler.fit_transform(X)
X_std[:,0]=(X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1]=(X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()
X_std[:,2]=(X_std[:,2]-X_std[:,2].mean())/X_std[:,2].std()
X_std[:,3]=(X_std[:,3]-X_std[:,3].mean())/X_std[:,3].std()
X_std[:,4]=(X_std[:,4]-X_std[:,4].mean())/X_std[:,4].std()
X_std[:,5]=(X_std[:,5]-X_std[:,5].mean())/X_std[:,5].std()
X_std[:,6]=(X_std[:,6]-X_std[:,6].mean())/X_std[:,6].std()
rng = np.random.RandomState(1)
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.4,random_state=rng)

dtree = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),
                          n_estimators=1000,learning_rate =0.2, random_state=rng)

X_train = X_train.astype(np.float)


#y_train = np.ndarray.flatten(y_train)
#y_train.astype(np.float)

dtree.fit(X_train,y_train)
y_train_pred = dtree.predict(X_train)
#for i in range(len(y_train_pred )):
#    y_train_pred[i]=2*y_train_pred[i]+5

y_test_pred = dtree.predict(X_test)
#for i in range(len(y_test_pred)):
#    y_test_pred[i]=2*y_test_pred[i]+5
df = X_test.copy()
df=pd.DataFrame(df)
df.to_csv("C:/Baseline") # save baseline for prediction
df['baseline']= y_test_pred
y_test_pred.astype(np.float)
y_train_pred.astype(np.float)
rmse_train = sqrt(mean_squared_error(y_train,y_train_pred))
rmse_test= sqrt( mean_squared_error(y_test,y_test_pred))
print('RMSE train: % .3f,test: %.3f' % (rmse_train, rmse_test))


plt.figure(1)
plt.subplot(211)
plt.title("Boosted Decision Tree Regression")
plt.scatter(y_train,y_train_pred-y_train, c='black', label="training samples")
#plt.scatter(y_train,y_train_pred, c='black', label="training samples")
plt.xlabel("Actual time")
plt.ylabel("Predicted time variation")
plt.legend()
plt.show()

plt.subplot(212)
plt.scatter(y_test,y_test_pred-y_test, c='red', label="testing samples")
#plt.scatter(y_test,y_test_pred, c='red', label="testing samples")
#plt.plot(X, y_2, c=y_train_pred"r", label="n_estimators=300", linewidth=2)
plt.xlabel("Actual time")
plt.ylabel("Predicted time variation")


plt.legend()
plt.show()

#export_graphviz(dtree,out_file='tree.dot',feature_names=['Vessel Bay','Vessel Row	Vessel',' Tier','Yard Location Updates','Volume','Standby_Hours','Restows'])
#shape, scale = 2., 2. # mean=4, std=2*sqrt(2)
#s = np.random.gamma(shape, scale, 1000)


