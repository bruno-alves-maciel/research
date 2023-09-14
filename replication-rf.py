# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:17:52 2023

@author: bruno
"""

import os
import pandas as pd
import numpy as np
import pingouin as pg
import pickle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import t
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import math
import statsmodels.api as sm

ROOT_DIR = os.getcwd() + r'/../'
MODEL = 'rf_09-06-23'

def setup_data(df_l = pd.read_csv(ROOT_DIR + r'output/clean_scaled.csv',index_col=0)):
    for c in df_l.columns:
        if type(df_l[c][0])==str:
            df_l[c] = pd.factorize(df_l[c])[0]
        elif (df_l[c][0] % 1 == 0):
            df_l[c] = pd.factorize(df_l[c])[0]
    X_l=df_l.drop(['LGB_id','weight','sexpart'],axis=1)
    y_l=df_l['LGB_id']
    return X_l,y_l, df_l
    

def train_rf_model(b_param = {'n_estimators':200, 'random_state':0, 'criterion': 'entropy', 'max_features':None, 'ccp_alpha':0.0}):

    #Train/test through LOGOCV
    logo = LeaveOneGroupOut()
    #loo.get_n_splits(X,y,df['sitecode'])
    logo.get_n_splits(X,y,groups=df['sitecode'])

    #clf = RandomForestClassifier(n_estimators=200, random_state=0, verbose=2, n_jobs=-1, class_weight='balanced', ccp_alpha=0.05174167)
    clf = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight='balanced', **b_param)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=df['sitecode'])

    scores = cross_validate(clf, X,y,cv = logo, groups=df['sitecode'])
    i = scores['test_score'].argmax()               #i = 12; 35
    #i = pd.unique(X['sitecode'])[scores['test_score']==max(scores['test_score'])]
    #score=[]
    #for k in i:
    #    X_tr = X[~(df['sitecode']==k)]
    #    X_te = X[df['sitecode']==k]
    #    y_tr = y[~(df['sitecode']==k)]
    #    y_te = y[df['sitecode']==k]
    #    clf.fit(X_tr,y_tr)
    #    score.append((clf.predict(X)==y).sum())
        
    X_tr = X[~(df['sitecode']==i)]
    X_te = X[df['sitecode']==i]
    y_tr = y[~(df['sitecode']==i)]
    y_te = y[df['sitecode']==i]
    clf.fit(X_tr,y_tr)
    
    '''
    path = clf.estimator.cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = list(ccp_alphas)
    impurities = list(impurities)
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[-300:-50], impurities[-300:-50], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    '''
    
    pickle.dump(clf, open(ROOT_DIR + r"models/"+MODEL+".mod", 'wb'))
    
    
def test_array(Xl,yl,dl, b_param = {'n_estimators':50, 'random_state':0, 'criterion': 'entropy', 'max_features':'sqrt', 'ccp_alpha':3.2e-5}):
    clf = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight='balanced', **b_param)
    mse=pd.DataFrame()
    complete = pd.DataFrame()
    prop = pd.DataFrame([[yl.mean()/len(yl),'Truth','USA','Complete',yl.mean()/len(yl)]])
    models = {}
    for s in pd.unique(dl['sitecode']):
        y_te = yl[dl['sitecode']==s]
        if len(y_te) == 0: continue
        y_tr = y[dl['sitecode']!=s]
        pro = y_te.mean()
        clf.fit(Xl[Xl['sitecode']!=s], y_tr)
        models[str(s)]=clf
        #General RF
        y_p = clf.predict_proba(Xl[Xl['sitecode']==s])
        com = pd.concat([Xl[Xl['sitecode']==s],pd.Series(y_p[:,1]),pd.Series(y_te)],axis=1)
        com['Model']='RF'
        com['State']=s
        com['Sex']='All'
        com['Age']='All'
        com['Race'] = 'All'
        complete = pd.concat([complete,com],axis=0)
        r = pd.DataFrame([(y_p.mean()-y_te.mean())**2,'RF','General',s])
        mse = pd.concat([mse,r.transpose()],axis=0)
        #General RF values
        r = pd.DataFrame([y_p.mean(),'RF',s,'General',y_te.mean()])
        prop = pd.concat([prop,r.transpose()],axis=0)
        #General Naive
        r = pd.DataFrame([(y_tr.mean()-y_te.mean())**2,'Naive','General',s])
        mse = pd.concat([mse,r.transpose()],axis=0)
        #General Naive Values
        r = pd.DataFrame([y_tr.mean(),'Naive',s,'General',y_te.mean()])
        prop = pd.concat([prop,r.transpose()],axis=0)
        r = pd.DataFrame([y_te.mean(),'Truth',s,'General',y_te.mean()])
        prop = pd.concat([prop,r.transpose()],axis=0)
        
        #Sex-based
        for x in pd.unique(dl['sex']):
            y_te = yl[(dl['sitecode']==s) & (dl['sex']==x)]
            if len(y_te) == 0: continue
            y_tr = y[(dl['sitecode']!=s) & (dl['sex']==x)]
            if len(y_te) == 0: continue
            pro = y_te.mean()
            clf.fit(X[(X['sitecode']!=s) & (dl['sex']==x)], y_tr)
            models[str(s)+'-'+str(x)]=clf
            #General RF
            y_p = clf.predict_proba(X[(X['sitecode']==s) & (dl['sex']==x)])
            com = pd.concat([X[X['sitecode']==s],pd.Series(y_p[:,1]),pd.Series(y_te)],axis=1)
            com['Model']='RF'
            com['State']=s
            com['Sex']=x
            com['Age']='All'
            com['Race'] = 'All'
            complete = pd.concat([complete,com],axis=0)
            r = pd.DataFrame([((y_p).mean()-y_te.mean())**2,'RF','x='+str(x),s])
            mse = pd.concat([mse,r.transpose()],axis=0)
            #General RF values
            r = pd.DataFrame([y_p.mean(),'RF',s,'x='+str(x),y_te.mean()])
            prop = pd.concat([prop,r.transpose()],axis=0)
            #General Naive
            r = pd.DataFrame([(y_tr.mean()-y_te.mean())**2,'Naive','x='+str(x),s])
            mse = pd.concat([mse,r.transpose()],axis=0)
            #General Naive Values
            r = pd.DataFrame([y_tr.mean(),'Naive',s,'x='+str(x),y_te.mean()])
            prop = pd.concat([prop,r.transpose()],axis=0)
            r = pd.DataFrame([y_te.mean(),'Truth',s,'x='+str(x),y_te.mean()])
            prop = pd.concat([prop,r.transpose()],axis=0)
            
            #Sex/age
            for a in pd.unique(dl['age']):
                y_te = yl[(dl['sitecode']==s) & (dl['sex']==x) & (dl['age']==a)]
                if len(y_te) == 0: continue
                y_tr = y[(dl['sitecode']!=s) & (dl['sex']==x) & (dl['age']==a)]
                pro = y_te.mean()
                clf.fit(X[(X['sitecode']!=s) & (dl['sex']==x) & (dl['age']==a)], y_tr)
                models[str(s)+'-'+str(x)+'-'+str(a)]=clf
                #General RF
                y_p = clf.predict_proba(X[(X['sitecode']==s) & (dl['sex']==x) & (dl['age']==a)])
                com = pd.concat([X[X['sitecode']==s],pd.Series(y_p[:,1]),pd.Series(y_te)],axis=1)
                com['Model']='RF'
                com['State']=s
                com['Sex']=x
                com['Age']=a
                com['Race'] = 'All'
                complete = pd.concat([complete,com],axis=0)
                r = pd.DataFrame([((y_p).mean()-y_te.mean())**2,'RF','x='+str(x)+r'/a='+str(a),s])
                mse = pd.concat([mse,r.transpose()],axis=0)
                #General RF values
                r = pd.DataFrame([y_p.mean(),'RF',s,'x='+str(x)+r'/a='+str(a),y_te.mean()])
                prop = pd.concat([prop,r.transpose()],axis=0)
                #General Naive
                r = pd.DataFrame([(y_tr.mean()-y_te.mean())**2,'Naive','x='+str(x)+r'/a='+str(a),s])
                mse = pd.concat([mse,r.transpose()],axis=0)
                #General Naive Values
                r = pd.DataFrame([y_tr.mean(),'Naive',s,'x='+str(x)+r'/a='+str(a),y_te.mean()])
                prop = pd.concat([prop,r.transpose()],axis=0)
                r = pd.DataFrame([y_te.mean(),'Truth',s,'x='+str(x)+r'/a='+str(a),y_te.mean()])
                prop = pd.concat([prop,r.transpose()],axis=0)
                
                #Sex/age/race
                for ra in pd.unique(dl['race4']):
                    y_te = yl[(dl['sitecode']==s) & (dl['sex']==x) & (dl['age']==a) & (dl['race4']==ra)]
                    if len(y_te) == 0: continue
                    y_tr = y_te = y[(dl['sitecode']!=s) & (dl['sex']==x) & (dl['age']==a) & (dl['race4']==ra)]
                    pro = y_te.mean()
                    clf.fit(X[(X['sitecode']!=s) & (dl['sex']==x) & (dl['age']==a) & (dl['race4']==ra)], y_tr)
                    models[str(s)+'-'+str(x)+'-'+str(a)+'-'+str(ra)]=clf
                    #General RF
                    y_p = clf.predict_proba(X[(X['sitecode']==s) & (dl['sex']==x) & (dl['age']==a) & (dl['race4']==ra)])
                    com = pd.concat([X[X['sitecode']==s],pd.Series(y_p[:,1]),pd.Series(y_te)],axis=1)
                    com['Model']='RF'
                    com['State']=s
                    com['Sex']=x
                    com['Age']=a
                    com['Race'] = ra
                    complete = pd.concat([complete,com],axis=0)
                    r = pd.DataFrame([((y_p).mean()-y_te.mean())**2,'RF','x='+str(x)+r'/a='+str(a)+r'/r='+str(ra),s])
                    mse = pd.concat([mse,r.transpose()],axis=0)
                    #General RF values
                    r = pd.DataFrame([y_p.mean(),'RF',s,'x='+str(x)+r'/a='+str(a)+r'/r='+str(ra),y_te.mean()])
                    prop = pd.concat([prop,r.transpose()],axis=0)
                    #General Naive
                    r = pd.DataFrame([(y_tr.mean()-y_te.mean())**2,'Naive','x='+str(x)+r'/a='+str(a)+r'/r='+str(ra),s])
                    mse = pd.concat([mse,r.transpose()],axis=0)
                    #General Naive Values
                    r = pd.DataFrame([y_tr.mean(),'Naive',s,'x='+str(x)+r'/a='+str(a)+r'/r='+str(ra),y_te.mean()])
                    prop = pd.concat([prop,r.transpose()],axis=0)
                    r = pd.DataFrame([y_te.mean(),'Truth',s,'x='+str(x)+r'/a='+str(a)+r'/r='+str(ra),y_te.mean()])
                    prop = pd.concat([prop,r.transpose()],axis=0)
                    
    mse.columns = ['MSE','Type','Segregation','Proof-state']
    mse.reset_index(drop=True,inplace=True)
    prop.reset_index(drop=True,inplace=True)
    prop.columns = ['Proportion','Type','State','Segregation','Truth']
    com.reset_index(drop=True,inplace=True)
    com.columns = X.columns+['Prediction','Real','Model','State','Sex','Age','Race']
    complete.to_csv(ROOT_DIR+'output/results.csv')
    mse[['MSE','Type']].groupby('Type').mean()
    pickle.dump(models, open(ROOT_DIR + r"models/"+MODEL+"-dict.mod", 'wb'))
    return mse, prop

    '''
    for s in pd.unique(dl['sitecode']):
        y_te = yl[dl['sitecode']==s]
        r = pd.DataFrame([[y_te.mean(),'Truth',s,'General']],columns = ['Proportion','Type','State','Segregation'])
        prop = pd.concat([prop,r],axis=0)
        for x in pd.unique(dl['sex']):
            y_te = yl[(dl['sitecode']==s) & (dl['sex']==x)]
            r = pd.DataFrame([[y_te.mean,'Truth',s,'x='+str(x)]],columns = ['Proportion','Type','State','Segregation'])
            prop = pd.concat([prop,r],axis=0)
            for a in pd.unique(dl['age']):
                y_te = yl[(dl['sitecode']==s) & (dl['sex']==x) & (dl['age']==a)]
                r = pd.DataFrame([[y_te.mean,'Truth',s,'x='+str(x)+r'/a='+str(a)]],columns = ['Proportion','Type','State','Segregation'])
                prop = pd.concat([prop,r],axis=0)
                for ra in pd.unique(dl['race4']):
                    y_te = yl[(dl['sitecode']==s) & (dl['sex']==x) & (dl['age']==a) & (dl['race4']==ra)]
                    r = pd.DataFrame([[y_te.mean,'Truth',s,'x='+str(x)+r'/a='+str(a)+r'/r='+str(ra)]],columns = ['Proportion','Type','State','Segregation'])
                    prop = pd.concat([prop,r],axis=0)
                    
    for s in pd.unique(prop['State']):
        for e in pd.unique(prop['Segregation']):
            for f in prop[(prop['State']==s) & (prop['Segregation']==e)]:
                f['Truth'] = prop.loc[(prop['State']==s) & (prop['Segregation']==e) & (prop['Type']=='Truth'),'Proportion']
    for k in prop[prop['Truth'].isna()].index:
        prop.loc[k,'Truth'] = prop.loc[(prop['Type']=='Truth') & (prop['State'] == prop.loc[k,'State']) & (prop['Segregation']==prop.loc[k,'Segregation']),'Proportion']
    '''
    
def train_grid_rf_model(Xl,yl,dl):

    #Train/test through LOGOCV
    logo = LeaveOneGroupOut()

    clf = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight='balanced')
    param_grid = {
        'n_estimators': [100,200],
        'max_features': [None,'sqrt'],              #Best: None
        'criterion' :['gini', 'entropy'],           #Best: gini
        'ccp_alpha' : [3.4e-5,5e-5]
    }

    CV_rf = GridSearchCV(estimator= clf,param_grid = param_grid, cv = logo)
    CV_rf.fit(Xl,yl,groups=dl['sitecode'])
    clf = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight='balanced', params = CV_rf.best_params_)
    pickle.dump(clf, open(ROOT_DIR + r"models/cv-"+MODEL+".mod", 'wb'))
    
def naive_train(df_l):
    logo = LeaveOneGroupOut()
    if len(pd.unique(df_l['sitecode']))<2:
        return df_l['LGB_id'].mean()
    sets = [i for i in logo.split(df_l.copy(),groups=df_l['sitecode'])]
    best_s = 100
    best_i = -1
    for s in sets:
        y_p = df_l.iloc[s[0],:]['LGB_id'].mean()
        y_r = df_l.iloc[s[1],:]['LGB_id'].mean()
        if (y_p-y_r)**2 < best_s:
            best_s = (y_p-y_r)**2
            best_i = s
    return df_l.iloc[best_i[0],:]['LGB_id'].mean()

def naive_state_estimation(df_l):
    return df_l[['sitecode','LGB_id']].groupby('sitecode').mean().median()

def naive_median_train(df_l):
    logo = LeaveOneGroupOut()
    if len(pd.unique(df_l['sitecode']))<2:
        return df_l['LGB_id'].mean()
    sets = [i for i in logo.split(df_l.copy(),groups=df_l['sitecode'])]
    best_s = 100
    best_i = -1
    for s in sets:
        y_p = df_l.iloc[s[0],:][['sitecode','LGB_id']].groupby('sitecode').mean().median()[0]
        y_r = df_l.iloc[s[1],:]['LGB_id'].mean()
        if (y_p-y_r)**2 < best_s:
            best_s = (y_p-y_r)**2
            best_i = s
    return df_l.iloc[best_i[0],:][['sitecode','LGB_id']].groupby('sitecode').mean().median()[0]
    
def calculate_metrics(X_l,y_l,clf,df_l):
    y_p = clf.predict(X_l)
    y_l.name = 'prediction'
    residual = (y_p!=y_l).sum()
    df_l['state_estimates'] = df_l['weight']*y_p
    df_l['state_ground'] = df_l['weight']*y_l
    state = pd.concat([df_l[['sitecode','state_estimates','state_ground']].groupby('sitecode').sum(),df_l[['sitecode','weight']].groupby('sitecode').sum()],axis=1)
    state.columns = ['estimate','ground','count']
    state['est_proportion'] = state['estimate']/state['count']
    state['gro_proportion'] = state['ground']/state['count']
    mean_res = residual/len(y_l)
    print(mean_res)
    deg_freedom = len(y_l)-1
    std_res = (((y_p!=y_l)**2).sum()/(deg_freedom-1))**(1/2)
    min_err, max_err = t.interval(.95,df=deg_freedom,loc=mean_res,scale=std_res)
    state['lower_bound']=state['est_proportion']+min_err
    state['upper_bound']=state['est_proportion']+max_err
    df_l = pd.concat([df_l,pd.Series(y_p, name='prediction')],axis=1)
    icc = pg.intraclass_corr(data=df_l, targets='prediction', raters='sitecode', ratings='LGB_id')
    return state, icc, df_l

def accuracy_figures(dl):
    STATE = "General"
    pred_proportion = dl['prediction'].sum()/len(dl)
    real_proportion = dl['LGB_id'].sum()/len(dl)
    data = pd.DataFrame([pd.Series(['Real','General',real_proportion]),pd.Series(['Prediction','General',pred_proportion])])
    data.columns = ['Type','X','Proportion']
    colors = {'Prediction':'red','Real':'green'}    
    fig = data.plot.scatter(x = 'X', y = 'Proportion', c=data['Type'].apply(lambda x: colors[x]))
    fig.figure.suptitle(STATE, fontsize=12, fontweight='bold', y=.97)
    
    #Per sex
    data = data.drop(['X'], axis = 1)
    pred_m_pro = dl.loc[dl['sex']==0,'prediction'].sum()/len(dl[dl['sex']==0])
    real_m_pro = dl.loc[dl['sex']==0,'LGB_id'].sum()/len(dl[dl['sex']==0])
    pred_f_pro = dl.loc[dl['sex']==1,'prediction'].sum()/len(dl[dl['sex']==1])
    real_f_pro = dl.loc[dl['sex']==1,'LGB_id'].sum()/len(dl[dl['sex']==1])
    pro = [real_m_pro]+[pred_m_pro]+[real_f_pro]+[pred_f_pro]
    data = pd.concat([data,data],axis=0)
    sex = ['Male']*2 + ['Female']*2
    data['Sex'] = sex
    data['Proportion'] = pro
    fig = data.plot.scatter(x = 'Sex', y = 'Proportion', c=data['Type'].apply(lambda x: colors[x]))
    fig.figure.suptitle(STATE, fontsize=12, fontweight='bold', y=.97)
    
    #include Age
    sex = ['Male', 'Female']
    age = pd.unique(dl['age'])
    l_age=[]
    for a in age: l_age = l_age + [a]*len(data)
    data = pd.concat([data]*len(age), axis=0)
    data['Age'] = l_age
    data.reset_index(drop = True, inplace=True)
    data['Category'] = '0'
    
    fields = {'Prediction':'prediction', 'Real':'LGB_id'}
    sex_dic = {'Male': 0, 'Female':1}
    for ty in pd.unique(data['Type']):
        for a in pd.unique(dl['age']):
            for s in pd.unique(sex):
                category = s[0]+'_'+ty
                data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == ty),'Category'] = category
                data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == ty),'Proportion'] = dl.loc[(dl['sex']==sex_dic[s]) & (dl['age']==a),fields[ty]].sum()/len(dl[(dl['sex']==sex_dic[s])  & (dl['age']==a)])
                #data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == 'Real'),'Proportion'] = dl.loc[(dl['sex']==s) & (dl['age']==a),'LGB_id'].sum()/len(dl[(dl['sex']==0)  & (dl['age']==0)])
    colors = cm.rainbow(np.linspace(0, 1, len(pd.unique(data['Category']))))
    
    fig, ax = plt.subplots()
    for c in pd.unique(data['Category']):
        dat = data[data['Category']==c]
        k='o'
        if c[2]=='R': k='d'
        plt.scatter(x = dat['Age'], y = dat['Proportion'], marker = k, label = c)
    ax.legend(loc="lower left")
    
    
    #Include race
    race_dic = {1: "White", 2: "Black/African", 3: "Hispanic/Latino", 4: "Others"}
    l_race=[]
    races = [race_dic[r] for r in race_dic]
    for a in races: l_race = l_race + [a]*len(data)
    data = pd.concat([data]*len(races), axis=0)
    data['Race'] = l_race
    data.reset_index(drop = True, inplace=True)
    race_rev_dic = {v: k for k,v in race_dic.items()}
    
    for ty in pd.unique(data['Type']):
        for a in pd.unique(dl['age']):
            for s in pd.unique(sex):
                for r in races:
                    category = r[:3]+"_"+s[0]+'_'+ty
                    data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == ty) & (data['Race'] == r),'Category'] = category
                    if len(dl[(dl['sex']==sex_dic[s])  & (dl['age']==a) & (dl['race4']==race_rev_dic[r])])==0:
                              data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == ty) & (data['Race'] == r),'Proportion'] = 0
                              continue
                    data.loc[(data['Sex']==s) & (data['Age']==a) & (data['Type'] == ty) & (data['Race'] == r),'Proportion'] = dl.loc[(dl['sex']==sex_dic[s]) & (dl['age']==a) & (dl['race4']==race_rev_dic[r]),fields[ty]].sum()/len(dl[(dl['sex']==sex_dic[s])  & (dl['age']==a) & (dl['race4']==race_rev_dic[r])])
    data = data.drop([dl['Race']=='Other'], axis = 0)
    data = data.drop((data[data['Race']=='Others']).index, axis = 0)
    colors = cm.rainbow(np.linspace(0, 1, len(pd.unique(data['Category']))))
    
    fig, ax = plt.subplots()
    for c in pd.unique(data['Category']):
        dat = data[data['Category']==c]
        k='o'
        if c[-1]=='l': k='d'
        plt.scatter(x = dat['Age'], y = dat['Proportion'], marker = k, label = c)
    ax.legend(loc="lower left")
    return data

def mse_figures(dl):
    dt = pd.DataFrame()
    #dl.iloc[dl.index % 3 == 1,1] #NN correction
    dt['Segregation'] = ['General','Sex','Sex, Age', 'Sex, Age, Race']
    dt['MSE'] = 0
    p = [dl.loc[(dl['Type']=='RF') & (dl['Segregation']=='General'),'MSE'].mean(), dl.loc[(dl['Type']=='RF') & ((dl['Segregation']=='x=0') | ((dl['Segregation']=='x=1'))),'MSE'].mean(), dl.loc[(dl['Type']=='RF') & (dl['Segregation'].apply(lambda k: len(k))==7),'MSE'].mean(), dl.loc[(dl['Type']=='RF') & (dl['Segregation'].apply(lambda k: len(k))>8),'MSE'].mean()]
    n = [dl.loc[(dl['Type']=='NN') & (dl['Segregation']=='General'),'MSE'].mean(), dl.loc[(dl['Type']=='NN') & ((dl['Segregation']=='x=0') | ((dl['Segregation']=='x=1'))),'MSE'].mean(), dl.loc[(dl['Type']=='NN') & (dl['Segregation'].apply(lambda k: len(k))==7),'MSE'].mean(), dl.loc[(dl['Type']=='NN') & (dl['Segregation'].apply(lambda k: len(k))>8),'MSE'].mean()]
    m = [dl.loc[(dl['Type']=='Naive') & (dl['Segregation']=='General'),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') & ((dl['Segregation']=='x=0') | ((dl['Segregation']=='x=1'))),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') & (dl['Segregation'].apply(lambda k: len(k))==7),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') & (dl['Segregation'].apply(lambda k: len(k))>8),'MSE'].mean()]
    dt['MSE'] = p
    dt['Type']="RF Estimate"
    dt = pd.concat([dt,dt,dt], axis = 0)
    dt.reset_index(drop=True, inplace=True)
    dt.iloc[len(m):(len(m)+len(n)),1] = m
    dt.iloc[len(m):(len(m)+len(n)),2] = 'Naive'
    dt.iloc[(len(m)+len(n)):,1] = n
    dt.iloc[(len(m)+len(n)):,2] = 'NN Estimate'
    
    colors = {'RF Estimate':'Red', 'Naive':'Green', 'NN Estimate':'Yellow'}
    #Print actual graph
    fig, ax = plt.subplots()
    plt.title('MSE between Estimated and Real LGB Proportions')
    plt.xlabel('Segregation Level')
    plt.ylabel('MSE')
    for c in pd.unique(dt['Type']):
        dat = dt[dt['Type']==c]
        plt.scatter(x = dat['Segregation'], y = dat['MSE'], label = c, color = colors[c])
    ax.legend(loc="upper left")
    
def residual_figures(dl):
    '''
    cat = [c for c in pd.unique(dl['Category']) if c[-1]=='n']
    data = dl.drop(dl[dl['Type']=='Real'].index, axis=0).drop(['Type'],axis=1)
    data['Category'] = data['Category'].apply(lambda x: x[:5])
    data.reset_index(drop = True, inplace=True)
    data['Sqr. Error'] = (data['Proportion'].values - dl.loc[dl['Type']=='Real','Proportion'].values)**2
    data['Sqr. Error'] = [(dl.loc[dl['Category']==c,'Proportion'].sum()-dl.loc[dl['Category']==(c[:5]+'_Real'),'Proportion'].sum())**2 for c in cat]
    '''
    dt = pd.DataFrame()
    dt['Segregation'] = ['General','Sex','Sex, Age', 'Sex, Age, Race', 'Sex, Age, Race, State']
    dt['MSE'] = 0
    dt2=dt.copy()
    p = [(dl['prediction'].sum()/len(dl) - dl['LGB_id'].sum()/len(dl))**2]
    m = [((naive_train(dl) - dl['LGB_id'].sum())/len(dl))**2]
    e = [((naive_median_train(dl) - dl['LGB_id'].sum())/len(dl))**2]
    # By sex
    k = dl[['sex','prediction','LGB_id']].groupby('sex').mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    k['Naive'] = [naive_train(dl[dl['sex']==d]) for d in pd.unique(dl['sex'])]
    k['Median'] = [naive_median_train(dl[dl['sex']==d]) for d in pd.unique(dl['sex'])]
    m.append(((k['Naive']-k['LGB_id'])**2).mean())
    e.append(((k['Median']-k['LGB_id'])**2).mean())
    # By Sex/Age
    k = dl[['sex','age','prediction','LGB_id']].groupby(['sex','age']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    k['Naive'] = [naive_train(dl[(dl['sex']==s) & (dl['age']==a)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age'])]
    k['Median'] = [naive_median_train(dl[(dl['sex']==s) & (dl['age']==a)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age'])]
    m.append(((k['Naive']-k['LGB_id'])**2).mean())
    e.append(((k['Median']-k['LGB_id'])**2).mean())
    # By Sex/Age/Race
    k = dl[['sex','age','race4','prediction','LGB_id']].groupby(['sex','age','race4']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    k['Naive'] = [naive_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age']) for r in pd.unique(dl['race4'])]
    k['Median'] = [naive_median_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age']) for r in pd.unique(dl['race4'])]
    m.append(((k['Naive']-k['LGB_id'])**2).mean())
    e.append(((k['Median']-k['LGB_id'])**2).mean())
    # By Sex/Age/Race/State
    k = dl[['sex','age','race4','sitecode','prediction','LGB_id']].groupby(['sex','age','race4','sitecode']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    k['Naive'] = [naive_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age']) for r in pd.unique(dl['race4']) for si in pd.unique(dl['sitecode']) if not math.isnan(naive_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r) & (dl['sitecode']==si)]))]
    m.append(((k['Naive']-k['LGB_id'])**2).mean())
    k['Median'] = [naive_median_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r)]) for s in pd.unique(dl['sex']) for a in pd.unique(dl['age']) for r in pd.unique(dl['race4']) for si in pd.unique(dl['sitecode']) if not math.isnan(naive_train(dl[(dl['sex']==s) & (dl['age']==a) & (dl['race4']==r) & (dl['sitecode']==si)]))]
    e.append(((k['Median']-k['LGB_id'])**2).mean())
    dt['MSE'] = p
    dt['Type']="ML Estimate"
    dt = pd.concat([dt,dt], axis = 0)
    dt.reset_index(drop=True, inplace=True)
    dt.iloc[len(m):,1] = m
    dt.iloc[len(m):,2] = 'Naive'
    dt2['MSE'] = e
    dt2['Type']="Median"
    dt=pd.concat([dt,dt2],axis=0)
    dt.reset_index(drop=True,inplace=True)
    
    colors = {'ML Estimate':'Red', 'Naive':'Green', 'Median':'Yellow'}
    #Print actual graph
    fig, ax = plt.subplots()
    plt.title('MSE between Estimated and Real LGB Proportions')
    for c in pd.unique(dt['Type']):
        dat = dt[dt['Type']==c]
        plt.scatter(x = dat['Segregation'], y = dat['MSE'], label = c, color = colors[c])
    ax.legend(loc="lower left")
    
    
    ##Starting by state
    dt = pd.DataFrame()
    dt['Segregation'] = ['General','State','State, Sex', 'State, Sex, Age', 'State, Sex, Age, Race']
    dt['MSE'] = 0
    p = [(dl['prediction'].sum()/len(dl) - dl['LGB_id'].sum()/len(dl))**2]
    m = [((naive_train(dl) - dl['LGB_id'].sum())/len(dl))**2]
    # By state
    k = dl[['sitecode','prediction','LGB_id']].groupby('sitecode').mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    k['Naive'] = [naive_train(dl[dl['state']==d]) for d in pd.unique(dl['sitecode'])]
    m.append(((k['Naive']-k['LGB_id'])**2).mean())
    k = dl[['sitecode','prediction','LGB_id']].groupby('sitecode').agg(pd.Series.mode)
    # By Sex/Age
    k = dl[['sex','sitecode','prediction','LGB_id']].groupby(['sex','sitecode']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    # By Sex/Age/Race
    k = dl[['sex','age','sitecode','prediction','LGB_id']].groupby(['sex','age','sitecode']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    # By Sex/Age/Race/State
    k = dl[['sex','age','race4','sitecode','prediction','LGB_id']].groupby(['sex','age','race4','sitecode']).mean()
    p.append(((k['prediction']-k['LGB_id'])**2).mean())
    dt['MSE'] = p
    
    #Print actual graph
    fig, ax = plt.subplots()
    plt.scatter(x = dt['Segregation'], y = dt['MSE'])
    
def icc_figure(dl):
    dt = dl[(dl['Type']!='Truth') & (dl['Segregation']=='General')]
    colors = {'RF':'Red', 'Naive':'Green', 'Truth':'Blue'}
    #Print actual graph
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5) 
    plt.title('Predicted proportions of LGB individuals by real proportions')
    plt.xlabel('Real proportions')
    plt.ylabel('Estimate proportions')
    dt['Proportion'] = pd.to_numeric(dt['Proportion'])
    #dt['Truth'] = dt['Truth'].apply(lambda x: pd.to_numeric(x.item()))
    dt['Truth'] = pd.to_numeric(dt['Truth'])
    #icc = pg.intraclass_corr(data=dt, targets='States', raters='Proportion', ratings='Type')
    box_style=dict(boxstyle='round', facecolor='blue', alpha=0.5)
    
    for c in pd.unique(dt['Type']):
        dat = dt[dt['Type']==c]
        plt.scatter(x = dat['Truth'], y = dat['Proportion'], label = c, color = colors[c])
        z = np.polyfit(dat['Truth'], dat['Proportion'], 1)
        p = np.poly1d(z)
        plt.plot([0,.15], p([0,.15]), color = colors[c],linestyle=(0,(5,5)))
        
    plt.plot(np.linspace(0,.15,34), np.linspace(0,.15,34), color = 'gray',linestyle='solid')
    ax.legend(loc="upper left")
    ax.set_xlim([0, .15])
    ax.set_ylim([0, .15])
    
    '''
    M = pd.concat([dt.loc[dt['Type']=='RF','Proportion'].reset_index(drop=True),dt.loc[dt['Type']=='Naive','Proportion'].reset_index(drop=True)],axis=1)
    M.columns = ['RF','Naive']
    xl = M.mean().mean()
    SST = ((M-xl)**2).sum().sum()
    SSBS = ((((M['RF']+M['Naive'])/2)-xl)**2).sum()
    SSBM = ((M.mean()-xl)**2).sum()
    SSWS = ((M['RF']-((M['RF']+M['Naive'])/2))**2+(M['Naive']-((M['RF']+M['Naive'])/2))**2).sum()
    SSWM = ((M['RF']-M.mean()[0])**2 + (M['Naive']-M.mean()[1])**2).sum()
    SSE = SST - SSBS - SSBM
    MST = SST/(len(M)*len(M.columns)-1)
    MSBM = SSBM/(len(M.columns)-1)
    MSBS = SSBS/(len(M)-1)
    MSWS = SSWS/len(M)/(len(M.columns)-1)
    MSWM = SSWM/len(M.columns)/(len(M)-1)
    MSE = SSE/(len(M)-1)/(len(M.columns)-1)
    ICC1 = (MSBS - MSWS)/(MSBS+(len(M.columns)-1)*MSWS)
    '''

#icc_figure(pd.read_csv(ROOT_DIR + r"output/prop.csv", index_col=0))
#mse_figures(pd.read_csv("C:/Users/bruno/Downloads/mse.csv", index_col=0))
X,y,df = setup_data()

mse, prop = test_array(X, y, df)
mse.to_csv(ROOT_DIR+r'/output/mse_test.csv')
prop.to_csv(ROOT_DIR+r'/output/prop_test.csv')

mse_figures(mse)

#train_rf_model()
#train_grid_rf_model(X,y,df)
clf = pickle.load(open(ROOT_DIR + r"models/"+MODEL+".mod", 'rb'))

state_t, icc_t, df_t = calculate_metrics(X, y, clf,df)

X_p,y_p,df_p = setup_data(pd.read_csv(ROOT_DIR + r"output/proof_scaled.csv", index_col=0))
state_p, icc_p, df_p = calculate_metrics(X_p,y_p,clf,df_p)

#data = accuracy_figures(df_t)
residual_figures(df_t)


#icc = pg.intraclass_corr(data=df_i, targets='', raters='judge', ratings='rating')
pickle.dump(clf, open(ROOT_DIR + r"models/rf.mod", 'wb'))

scores = []
for s in pd.unique(df['sitecode']):
    X_te,y_te = X[df['sitecode']==s], y[df['sitecode']==s]
    y_p = clf.predict(X_te)
    scores.append((y_te==y_p).sum()/len(y_te))




##General test
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)