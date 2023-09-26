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
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import warnings

ROOT_DIR = os.getcwd() + r'/../'
MODEL = '09-26-23'
warnings.filterwarnings("ignore")

def setup_data(df_l = pd.read_csv(ROOT_DIR + r'output/clean_scaled.csv',index_col=0)):
    for c in df_l.columns:
        if type(df_l[c][0])==str:
            df_l[c] = pd.factorize(df_l[c])[0]
        elif (df_l[c][0] % 1 == 0):
            df_l[c] = pd.factorize(df_l[c])[0]
    X_l=df_l.drop(['LGB_id','weight','sexpart'],axis=1)
    y_l=df_l['LGB_id']
    return X_l,y_l, df_l
    

def reduce_dimensions(dl:pd.DataFrame):
    #return dl
    pca = PCA(.95)
    pca.fit(dl)
    dl = pca.transform(dl)
    '''
    print(pca.n_components_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    plt.close()
    '''
    return dl
    

def reduced_train(naive:bool = False, 
                  clf = RandomForestClassifier(verbose=0, n_jobs=-1, class_weight='balanced',n_estimators=50, random_state=0, criterion='entropy'),
                  Xl: pd.DataFrame=None,
                  yl: pd.DataFrame()=None):
    try:
        if naive:
            y_tr = yl.sample(frac=.75)
            return y_tr.mean(), yl.drop(y_tr.index,axis=0).mean()
        Xl = reduce_dimensions(Xl)
        X_tr, X_te, y_tr, y_te = train_test_split(Xl, yl, stratify=yl)
        clf.fit(X_tr,y_tr)
        y_p = clf.predict_proba(X_te)
        y_p = [x[1] for x in y_p]
        
        #General proportion
        pred = np.mean(y_p)
        truth = y_te.mean()
        return pred,truth
    except:
        return 0,0

def logo_model_test(Xl:pd.DataFrame,
                       yl:pd.DataFrame, 
                       clf=RandomForestClassifier(verbose=0, n_jobs=-1, class_weight='balanced',n_estimators=50, random_state=0, criterion='entropy'),
                       ty: str = 'RF', 
                       naive:bool = False):
    #General:
    compare = pd.DataFrame()
    MSE = []
    preds = []
    truths = []
    sex = []
    age=[]
    race=[]
    state = []
    pop = []
    typ=[]
    auROC=[]
    ICCs = []
    for s in pd.unique(Xl['sitecode']):
        dt = Xl
        dat=yl
        tr = dt['sitecode']!=s
        dt = reduce_dimensions(dt)
        X_tr = dt[tr]
        X_te = dt[~tr]
        y_tr = dat[tr]
        y_te = dat[~tr]
        if naive:
            pred = y_tr.mean()
        else:
            try:
                clf.fit(X_tr,y_tr)
                y_p = clf.predict_proba(X_te)
                y_p = [x[1] for x in y_p]
                aur = roc_auc_score(y_te, pd.Series(y_p))
                pred=np.mean(y_p)
            except:
                pred=0
                aur=0
        truth = y_te.mean()
        MSE.append((pred-truth)**2)
        preds.append(pred)
        truths.append(truth)
        auROC.append(aur)
        sex.append('All')
        age.append('All')
        race.append('All')
        state.append(s)
        pop.append(len(X_te))
        typ.append(ty)
        for x in pd.unique(Xl['sex']):
            dt = Xl[Xl['sex']==x]
            dat = yl[Xl['sex']==x]
            tr = dt['sitecode']!=s
            dt = reduce_dimensions(dt)
            X_tr = dt[tr]
            X_te = dt[~tr]
            y_tr = dat[tr]
            y_te = dat[~tr]
            if naive:
                pred = y_tr.mean()
            else:
                try:
                    clf.fit(X_tr,y_tr)
                    y_p = clf.predict_proba(X_te)
                    y_p = [x[1] for x in y_p]
                    pred=np.mean(y_p)
                    aur = roc_auc_score(y_te, pd.Series(y_p))
                except:
                    pred=0
                    aur=0
            truth = y_te.mean()
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            auROC.append(aur)
            sex.append(x)
            age.append('All')
            race.append('All')
            state.append(s)
            pop.append(len(X_te))
            typ.append(ty)
        for r in pd.unique(Xl['race4']):
            dt = Xl[Xl['race4']==r]
            dat = yl[Xl['race4']==r]
            tr = dt['sitecode']!=s
            dt = reduce_dimensions(dt)
            X_tr = dt[tr]
            X_te = dt[~tr]
            y_tr = dat[tr]
            y_te = dat[~tr]
            if naive:
                pred = y_tr.mean()
            else:
                try:
                    clf.fit(X_tr,y_tr)
                    y_p = clf.predict_proba(X_te)
                    y_p = [x[1] for x in y_p]
                    pred=np.mean(y_p)
                    aur = roc_auc_score(y_te, pd.Series(y_p))
                except:
                    pred=0
                    aur=0
            truth = y_te.mean()
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            auROC.append(aur)
            sex.append('All')
            age.append('All')
            race.append(r)
            state.append(s)
            pop.append(len(X_te))
            typ.append(ty)
            for x in pd.unique(Xl['sex']):
                dt = Xl[(Xl['race4']==r) & (Xl['sex']==x)]
                dat = yl[(Xl['race4']==r) & (Xl['sex']==x)]
                tr = dt['sitecode']!=s
                dt = reduce_dimensions(dt)
                X_tr = dt[tr]
                X_te = dt[~tr]
                y_tr = dat[tr]
                y_te = dat[~tr]
                if naive:
                    pred = y_tr.mean()
                else:
                    try:
                        clf.fit(X_tr,y_tr)
                        y_p = clf.predict_proba(X_te)
                        y_p = [x[1] for x in y_p]
                        pred=np.mean(y_p)
                        aur = roc_auc_score(y_te, pd.Series(y_p))
                    except:
                        pred=0
                        aur=0
                truth = y_te.mean()
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                auROC.append(aur)
                sex.append(x)
                age.append('All')
                race.append(r)
                state.append(s)
                pop.append(len(X_te))
                typ.append(ty)
                for a in pd.unique(Xl['age']):
                    dt = Xl[(Xl['race4']==r) & (Xl['sex']==x) & (Xl['age']==a)]
                    dat = yl[(Xl['race4']==r) & (Xl['sex']==x) & (Xl['age']==a)]
                    tr = dt['sitecode']!=s
                    dt = reduce_dimensions(dt)
                    X_tr = dt[tr]
                    X_te = dt[~tr]
                    y_tr = dat[tr]
                    y_te = dat[~tr]
                    if naive:
                        pred = y_tr.mean()
                    else:
                        try:
                            clf.fit(X_tr,y_tr)
                            y_p = clf.predict_proba(X_te)
                            y_p = [x[1] for x in y_p]
                            pred=np.mean(y_p)
                            aur = roc_auc_score(y_te, pd.Series(y_p))
                        except:
                            pred=0
                            aur=0
                    truth = y_te.mean()
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    auROC.append(aur)
                    sex.append(x)
                    age.append(a)
                    race.append(r)
                    state.append(s)
                    pop.append(len(X_te))
                    typ.append(ty)
                    
    #Compile results
    compare['Proportion'] = preds
    compare['Truth'] = truths
    compare['MSE'] = MSE
    compare['Population'] = pop
    compare['AuR0C'] = auROC
    compare['Sex'] = sex
    compare['Age'] = age
    compare['Race'] = race
    compare['State'] = state
    compare['Type'] = typ
    
    compare.to_csv(ROOT_DIR+r'output/compare-'+ty+'.csv')
    return compare

def model_test(Xl:pd.DataFrame,yl:pd.DataFrame, clf=RandomForestClassifier(verbose=0, n_jobs=-1, class_weight='balanced',n_estimators=50, random_state=0, criterion='entropy'),ty: str = 'RF', naive:bool = False):
    compare = pd.DataFrame()
    #General
    pred,truth = reduced_train(naive,clf,Xl,yl)
    MSE = [(pred-truth)**2]
    preds = [pred]
    truths = [truth]
    sex = ['All']
    age=['All']
    race=['All']
    state = ['All']
    pop = [len(yl)]
    typ=[ty]
    
    for s in pd.unique(Xl['sitecode']):
        dt = Xl[Xl['sitecode']==s]
        dat = yl[Xl['sitecode']==s]
        pred,truth = reduced_train(naive,clf,Xl=dt,yl = dat)
        MSE.append((pred-truth)**2)
        preds.append(pred)
        truths.append(truth)
        sex.append('All')
        age.append('All')
        race.append('All')
        state.append(s)
        pop.append(len(dt))
        typ.append(ty)
        for x in pd.unique(Xl['sex']):
            dt = Xl[(Xl['sitecode']==s)&(Xl['sex']==x)]
            dat = yl[(Xl['sitecode']==s)&(Xl['sex']==x)]
            pred,truth = reduced_train(naive,clf,dt,dat)
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            sex.append(x)
            age.append('All')
            race.append('All')
            state.append(s)
            pop.append(len(dt))
            typ.append(ty)
            for r in pd.unique(Xl['race4']):
                dt = Xl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r)]
                dat = yl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r)]
                pred,truth = reduced_train(naive,clf,dt,dat)
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append(x)
                age.append('All')
                race.append(r)
                state.append(s)
                pop.append(len(dt))
                typ.append(ty)
                for a in pd.unique(Xl['age']):
                    dt = Xl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r) & (Xl['age']==a)]
                    dat = yl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r) & (Xl['age']==a)]
                    pred,truth = reduced_train(naive,clf,dt,dat)
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    sex.append(x)
                    age.append(a)
                    race.append(r)
                    state.append(s)
                    pop.append(len(dt))
                    typ.append(ty)
                    
        for r in pd.unique(Xl['race4']):
            dt = Xl[(Xl['sitecode']==s)&(Xl['race4']==r)]
            dat = yl[(Xl['sitecode']==s)&(Xl['race4']==r)]
            pred,truth = reduced_train(naive,clf,dt,dat)
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            sex.append('All')
            age.append('All')
            race.append(r)
            state.append(s)
            pop.append(len(dt))
            typ.append(ty)
            for x in pd.unique(Xl['sex']):
                dt = Xl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r)]
                dat = yl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['race4']==r)]
                pred,truth = reduced_train(naive,clf,dt,dat)
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append(x)
                age.append('All')
                race.append(r)
                state.append(s)
                pop.append(len(dt))
                typ.append(ty)
                for a in pd.unique(Xl['age']):
                    dt = Xl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['sitecode']==s) & (Xl['age']==a)]
                    dat = yl[(Xl['sitecode']==s)&(Xl['sex']==x) & (Xl['sitecode']==s) & (Xl['age']==a)]
                    pred,truth = reduced_train(naive,clf,dt,dat)
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    sex.append(x)
                    age.append(a)
                    race.append(r)
                    state.append(s)
                    pop.append(len(dt))
                    typ.append(ty)
    '''    
    for st in pd.unique(Xl['sitecode']):
        o=Xl
        Xl=Xl[Xl['sitecode']==st]
        #Sex-segregated;
        for x in pd.unique(Xl['sex']):
            pred,truth = reduced_train(naive,clf,Xl[Xl['sex']==x],yl[Xl['sex']==x])
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            sex.append(x)
            age.append('All')
            race.append('All')
            state.append(st)
            pop.append(len(yl[Xl['sex']==x]))
            typ.append(ty)
            #Age-segregated
            for a in pd.unique(Xl['age']):
                pred,truth = reduced_train(naive,clf,Xl[(Xl['sex']==x) & (Xl['age']==a)],yl[(Xl['sex']==x) & (Xl['age']==a)])
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append(x)
                age.append(a)
                race.append('All')
                state.append(st)
                pop.append(len(yl[(Xl['sex']==x) & (Xl['age']==a)]))
                typ.append(ty)
                #Race-segregated
                for ra in pd.unique(Xl['race4']):
                    pred,truth = reduced_train(naive,clf,Xl[(Xl['sex']==x) & (Xl['age']==a) & (Xl['race4']==ra)],yl[(Xl['sex']==x) & (Xl['age']==a) & (Xl['race4']==ra)])
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    sex.append(x)
                    age.append(a)
                    race.append(ra)
                    state.append(st)
                    pop.append(len(yl[(Xl['sex']==x) & (Xl['age']==a) & (Xl['race4']==ra)]))
                    typ.append(ty)
                        
        #Race-segregated;
        for x in pd.unique(Xl['race4']):
            pred,truth = reduced_train(naive,clf,Xl[Xl['race4']==x],yl[Xl['race4']==x])
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            sex.append('All')
            age.append('All')
            race.append(x)
            state.append('All')
            pop.append(len(yl[Xl['race4']==x]))
            typ.append(ty)
            #Age-segregated
            for a in pd.unique(Xl['age']):
                pred,truth = reduced_train(naive,clf,Xl[(Xl['race4']==x) & (Xl['age']==a)],yl[(Xl['race4']==x) & (Xl['age']==a)])
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append('All')
                age.append(a)
                race.append(x)
                state.append('All')
                pop.append(len(yl[(Xl['race4']==x) & (Xl['age']==a)]))
                typ.append(ty)
                #Sex-segregated
                for ra in pd.unique(Xl['sex']):
                    pred,truth = reduced_train(naive,clf,Xl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x)],yl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x)])
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    sex.append(ra)
                    age.append(a)
                    race.append(x)
                    state.append('All')
                    pop.append(len(yl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x)]))
                    typ.append(ty)
                    #State segregated
                    for s in pd.unique(Xl['sitecode']):
                        pred,truth = reduced_train(naive,clf,Xl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x) & (Xl['sitecode']==s)],yl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x) & (Xl['sitecode']==s)])
                        MSE.append((pred-truth)**2)
                        preds.append(pred)
                        truths.append(truth)
                        sex.append(ra)
                        age.append(a)
                        race.append(x)
                        state.append(s)
                        pop.append(len(yl[(Xl['sex']==ra) & (Xl['age']==a) & (Xl['race4']==x) & (Xl['sitecode']==s)]))
                        typ.append(ty)
        
        sequence=['sitecode','sex','age','race']
        k=0
        xt=Xl
        yt=y
        #State-segregated;
        for i in range(len(sequence)):
            ...#xt[sitecode+] = 'All'
            for j in pd.unique(xt[sequence[i]]):
                xt1=xt[xt[i]==j]
                ... #xt[sex+] = 'All'
                k = k+1
                if len(sequence)-k > 0:
                    for i2 in range(k,len(sequence)):
                        for j2 in pd.unique(xt1[sequence[i2]]):
                            xt2=xt1[xt1[sequence[i2]]==j2]
                            ... #xt[age+]='All'
                            k=k+1
                            if k < len(sequence):
                                for i3 in range(k,len(sequence)):
                                    for j3 in pd.unique(xt2[sequence[i3]]):
                                        xt3=xt2[xt2[sequence[i3]]==j3]
                                        ... #xt[race] = 'All'
                                        k=k+1
                                        if k < len(sequence):
                                            for i4 in range(k,len(sequence)):
                                                for j4 in pd.unique(xt3[sequence[i4]]):
                                                    xt4=xt3[xt3[i4]==j4]
                                                    ... #Different
                                                    k=k+1
                                                    if k == len(sequence):
                                                        continue
    '''                                        
    '''                                      
            xt=[Xl]
            i_l = []
            j_l = []
            ranges = [range(k,len(sequence)) for k in range(len(sequence))]
            k=0
            while k<len(sequence):
                i_l.append(range(k,len(sequence)))
                for i in i_l:
                    j_l.append(pd.unique(xt[k][sequence[i]]))
                    for j in pd.unique(xt[k][sequence[i]]):
                        for l in xt:
                            xt.append(l[l[sequence[i]==j]])
                        ...
                k=k+1
                ...
                k=k+1
                    
                                                    
                                                    
                        
            for j in pd.unique(xt[i]):
                xt_s=xt[xt[i]==j]
                yt_s=yt[xt[i]==j]
                pred,truth = reduced_train(naive,clf,xt_s,yt_s)
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append(pd.unique(xt_s['sex']).tolist())
                age.append((pd.unique(xt_s['age'])).tolist())
                race.append((pd.unique(xt_s['race4'])).tolist())
                state.append(pd.unique(xt_s['race4']).tolist())
                pop.append(len(yt_s))
                typ.append(ty)
                for i2
            xt=xt_s
            yt=yt_s
    '''
    '''
            dt = dt[]
            pred,truth = reduced_train(naive,clf,Xl[Xl[i]==x],yl[Xl[i]==x])
            MSE.append((pred-truth)**2)
            preds.append(pred)
            truths.append(truth)
            sex.append('All')
            age.append('All')
            race.append('All')
            state.append(x)
            pop.append(len(yl[Xl['sitecode']==x]))
            typ.append(ty)
            #Age-segregated
            for a in pd.unique(Xl['age']):
                pred,truth = reduced_train(naive,clf,Xl[(Xl['sitecode']==x) & (Xl['age']==a)],yl[(Xl['sitecode']==x) & (Xl['age']==a)])
                MSE.append((pred-truth)**2)
                preds.append(pred)
                truths.append(truth)
                sex.append("All")
                age.append(a)
                race.append('All')
                state.append(x)
                pop.append(len(yl[(Xl['sitecode']==x) & (Xl['age']==a)]))
                typ.append(ty)
                #Race-segregated
                for ra in pd.unique(Xl['race4']):
                    pred,truth = reduced_train(naive,clf,Xl[(Xl['sitecode']==x) & (Xl['age']==a) & (Xl['race4']==ra)],yl[(Xl['sitecode']==x) & (Xl['age']==a) & (Xl['race4']==ra)])
                    MSE.append((pred-truth)**2)
                    preds.append(pred)
                    truths.append(truth)
                    sex.append("All")
                    age.append(a)
                    race.append(ra)
                    state.append(x)
                    pop.append(len(yl[(Xl['sitecode']==x) & (Xl['age']==a) & (Xl['race4']==ra)]))
                    typ.append(ty)
                    #State segregated
                    for s in pd.unique(Xl['sex']):
                        pred,truth = reduced_train(naive,clf,Xl[(Xl['sex']==s) & (Xl['age']==a) & (Xl['race4']==ra) & (Xl['sitecode']==x)],yl[(Xl['sex']==s) & (Xl['age']==a) & (Xl['race4']==ra) & (Xl['sitecode']==x)])
                        MSE.append((pred-truth)**2)
                        preds.append(pred)
                        truths.append(truth)
                        sex.append(s)
                        age.append(a)
                        race.append(ra)
                        state.append(x)
                        pop.append(len(yl[(Xl['sex']==s) & (Xl['age']==a) & (Xl['race4']==ra) & (Xl['sitecode']==x)]))
                        typ.append(ty)
                        '''
                        
    #Compile results
    compare['Proportion'] = preds
    compare['Truth'] = truths
    compare['MSE'] = MSE
    compare['Population'] = pop
    compare['Sex'] = sex
    compare['Age'] = age
    compare['Race'] = race
    compare['State'] = state
    compare['Type'] = typ
    
    compare.to_csv(ROOT_DIR+r'output/compare-'+ty+'.csv')
    return compare


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
    auROC = roc_auc_score(y_l,y_p)
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
    sex_dic = {'Male': 0, 'Female':1, '0':'Male','1':'Female'}
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
    dt['Segregation'] = ['None','Sex', 'Race', 'Sex, Race', 'Sex, Age, Race']
    dt['MSE'] = 0
    p = [dl.loc[(dl['Type']=='RF') & (dl['Sex']=='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='RF') & (dl['Sex']!='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='RF') & (dl['Race']!='All') & (dl['Sex']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='RF') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']=='All'),'MSE'].mean(),dl.loc[(dl['Type']=='RF') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']!='All'),'MSE'].mean()]
    n = [dl.loc[(dl['Type']=='NN') & (dl['Sex']=='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='NN') & (dl['Sex']!='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='NN') & (dl['Race']!='All') & (dl['Sex']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='NN') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']=='All'),'MSE'].mean(),dl.loc[(dl['Type']=='NN') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']!='All'),'MSE'].mean()]
    m = [dl.loc[(dl['Type']=='Naive') & (dl['Sex']=='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') & (dl['Sex']!='All') & (dl['Race']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') & (dl['Race']!='All') & (dl['Sex']=='All'),'MSE'].mean(), dl.loc[(dl['Type']=='Naive') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']=='All'),'MSE'].mean(),dl.loc[(dl['Type']=='Naive') &  (dl['Sex']!='All') & (dl['Race']!='All') & (dl['Age']!='All'),'MSE'].mean()]
    dt['MSE'] = p
    dt['Type']="RF Estimate"
    dt = pd.concat([dt,dt,dt], axis = 0)
    dt.reset_index(drop=True, inplace=True)
    dt.iloc[len(m):(len(m)+len(n)),1] = m
    dt.iloc[len(m):(len(m)+len(n)),2] = 'Naive'
    dt.iloc[(len(m)+len(n)):,1] = n
    dt.iloc[(len(m)+len(n)):,2] = 'NN Estimate'
    
    colors = {'RF Estimate':'Red', 'Naive':'Green', 'NN Estimate':'Blue'}
    #Print actual graph
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6) 
    ax.set_title('MSE between Estimated and Real LGB Proportions', fontsize=20)
    ax.set_xlabel('Segregation Level', fontsize=16)
    ax.set_ylabel('MSE',fontsize=16)
    for c in pd.unique(dt['Type']):
        dat = dt[dt['Type']==c]
        plt.scatter(x = dat['Segregation'], y = dat['MSE'], label = c, color = colors[c])
    ax.legend(loc="upper left",fontsize=14)
    fig.savefig(ROOT_DIR+ 'figure/MSE.png')
    plt.close()
    
def calculate_pvalues(dl):
    dfcols = pd.DataFrame(columns=dl.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in dl.columns:
        for c in dl.columns:
            tmp = dl[dl[r].notnull() & dl[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues
    
def analyze_states(dl: pd.DataFrame()):
    prri= pd.read_csv(ROOT_DIR+r'raw_data_2019/PRRI.csv', index_col=0)
    for i in dl.index:
        for c in prri.columns:
            dl.loc[i,c] = prri.loc[state_dic[i[0]],c]
    correl = dl.corr()['Proportion'].drop(['Proportion'],axis=0)
    sns.heatmap(dl.corr(method='pearson', min_periods=1))
    plt.show()
    dt = dl.drop(['OpposeSSM',"Don't Know / RefusedSSM", 'OpposeProt', "Don't Know / RefusedProt", 'OpposeRel', "Don't Know / RefusedRel"],axis=1)
    
    reg = LinearRegression().fit(dl.drop('Proportion',axis=1), dl['Proportion'])
    reg.score(dl.drop('Proportion',axis=1), dl['Proportion'])
    
state_dic = {'AL':'Alabama', 'AR':'Arkansas', 'AZB':'Arizona', 'CA':'California', 'CO':'Colorado', 'CT':'Connecticut', 'FL':'Florida', 'HI':'Hawaii', 'IA':'Iowa', 'IL':'Illinois', 'KY':'Kentucky', 'MD':'Maryland', 'ME':'Maine', 'MI':'Michigan', 'MO':'Missouri', 'MS':'Mississippi', 'NC':'North Carolina', 'ND':'North Dakota', 'NE':'Nebraska', 'NH':'New Hampshire', 'NJ':'New Jersey', 'NM':'New Mexico', 'NV':'Nevada', 'NY':'New York', 'OK':'Oklahoma', 'PA':'Pennsylvania', 'RI':'Rhode Island', 'SC':'South Carolina', 'TX':'Texas', 'USA':'National', 'UT':'Utah', 'VA':'Virginia', 'VT':'Vermont', 'WI':'Wisconsin', 'WV':'West Virginia'}
sex_dic = {0:'Male', 1:'Female','All':'All', '0':'Male','1':'Female'}
age_dic = {0:'12y', 1:'13y',2:'14y',3:'15y',4:'16y',5:'17y',6:'18y','All':'All','0':'12y', '1':'13y','2':'14y','3':'15y','4':'16y','5':'17y','6':'18y'}
race_dic = {0:'White',1:'Black',2:'Latino',3:'Other','All':'All','0':'White','1':'Black','2':'Latino','3':'Other',}    
    
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
    
def icc_metrics(dl: pd.DataFrame):
    dl = dl[dl['State']!='USA']
    dl['Proportion'] = pd.to_numeric(dl['Proportion'])
    dl['Truth'] = pd.to_numeric(dl['Truth'])
    icc_df = pd.DataFrame()
    for s in pd.unique(dl['Sex']):
        for a in pd.unique(dl.loc[dl['Sex']==s,'Age']):
            for r in pd.unique(dl.loc[(dl['Sex']==s) & (dl['Age']==a),'Race']):
                for c in pd.unique(dl['Type']):
                    if c=='Truth': continue
                    dt = dl[(dl['Sex']==s) & (dl['Age']==a) & (dl['Race']==r)]
                    dt = dt[(dt['Type']==c) | (dt['Type']=='Truth')]
                    icc = pg.intraclass_corr(data=dt, targets='State', raters='Type', ratings='Proportion')
                    row = pd.DataFrame([[s,a,r,c]+icc.iloc[5,:].tolist()])
                    icc_df = pd.concat([icc_df,row],axis=0)
                    
    icc_df.columns = ['Sex','Age','Race','Method']+icc.columns.tolist()
    return icc_df
                
    
def all_icc_figures(dl):
    dl['Proportion'] = pd.to_numeric(dl['Proportion'])
    dl['Truth'] = pd.to_numeric(dl['Truth'])
    for s in pd.unique(dl['Sex']):
        for r in pd.unique(dl.loc[dl['Sex']==s,'Race']):
            for a in pd.unique(dl.loc[(dl['Sex']==s) & (dl['Race']==r),'Age']):
                dt = dl[(dl['Sex']==s) & (dl['Age']==a) & (dl['Race']==r)]
                colors = {'RF':'Red', 'Naive':'Green', 'Truth':'Blue', 'NN':'Blue'}
                #Print actual graph
                fig, ax = plt.subplots()
                fig.set_size_inches(8,8) 
                ax.set_title('Predicted proportions of LGB individuals \n by real proportions \nSex = '+sex_dic[s]+'   Age = '+age_dic[a]+'   Race = '+race_dic[r]+'   N = '+str(int(dt['Population'].median())), fontsize=20)
                #plt.subtitle('Sex='+str(s)+', Age='+str(a)+', Race='+str(r))
                ax.set_xlabel('Real proportions',fontsize=16)
                ax.set_ylabel('Estimate proportions',fontsize=16)
                
                box_style=dict(boxstyle='round', facecolor='blue', alpha=0.5)
                
                for c in pd.unique(dt['Type']):
                    if c=='Truth': continue
                    dat = dt[dt['Type']==c]
                    plt.scatter(x = dat['Truth'], y = dat['Proportion'], label = c, color = colors[c])
                    z = np.polyfit(dat['Truth'], dat['Proportion'], 1)
                    p = np.poly1d(z)
                    plt.plot([0,max(dt['Truth'])*1.05], p([0,max(dt['Truth'])*1.05]), color = colors[c],linestyle=(0,(5,5)))
                    
                plt.plot(np.linspace(0,max(dt['Truth'])*1.05,34), np.linspace(0,max(dt['Truth'])*1.05,34), color = 'gray',linestyle='solid')
                ax.legend(loc="upper left",fontsize=14)
                ax.set_xlim([0, max(dt['Truth']*1.05)])
                ax.set_ylim([0, max(dt['Truth']*1.05)])
                fig.savefig(ROOT_DIR+ 'figure/icc/Sex='+str(s)+'-Age='+str(a)+'-Race='+str(r)+'.png')
                plt.close()

def icc_figure(dl):
    #dt = dl[(dl['Type']!='Truth') & (dl['Segregation']=='General')]
    dt = dl[(dl['Type']!='Truth') & (dl['State']!='USA') & (dl['Sex'] == 'Complete')]
    colors = {'RF':'Red', 'Naive':'Green', 'Truth':'Blue', 'NN':'Yellow'}
    #Print actual graph
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5) 
    plt.title('Predicted proportions of LGB individuals by real proportions')
    plt.xlabel('Real proportions')
    plt.ylabel('Estimate proportions')
    dt['Proportion'] = pd.to_numeric(dt['Proportion'])
    #dt['Truth'] = dt['Truth'].apply(lambda x: pd.to_numeric(x.item()))
    dt['Truth'] = pd.to_numeric(dt['Truth'])
    #icc = pg.intraclass_corr(data=dt, targets='State', raters='Type', ratings='Proportion')
    #   Type              Description       ICC  ...  df2      pval           CI95%
    #5  ICC3k     Average fixed raters  0.688728  ...   33  0.000598    [0.38, 0.84]
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


#all_icc_figures(pd.read_csv(ROOT_DIR + r"output/cl-prop.csv", index_col=0))
#icc_df = icc_metrics(pd.read_csv(ROOT_DIR + r"output/cl-prop.csv", index_col=0))
#mse_figures(pd.read_csv(ROOT_DIR+r"output/cl-mse.csv", index_col=0))
#analyze_states(pd.read_csv(ROOT_DIR+'output/state-prop.csv',index_col=0))
X,y,df = setup_data()
#com = logo_model_test(Xl = X, yl = y, naive=True, ty='Naive')
#com = model_test(Xl = X, yl = y,naive = True,ty='Naive')
com=pd.read_csv(ROOT_DIR+r'output/compare-Naive.csv',index_col=0)
com = com.drop(com[com['Population']==0].index, axis=0)
com = com.dropna()
com = pd.concat([com,logo_model_test(Xl=X,yl=y)], axis = 0)
#com = pd.concat([com,model_test(Xl=X,yl=y)], axis = 0)
#com = pd.concat([com,pd.read_csv(ROOT_DIR+r'output/compare-RF.csv',index_col=0)], axis = 0)
com = pd.concat([com,logo_model_test(Xl=X,yl=y,clf=MLPClassifier(hidden_layer_sizes=(30,15), max_iter=500, activation='logistic', random_state=0, verbose=False),ty='NN')], axis=0)
com = com.drop(com[com['Population']==0].index, axis=0)
com = com.dropna()


mse, prop = test_array(X, y, df)
#mse.to_csv(ROOT_DIR+r'/output/mse_test.csv')
#prop.to_csv(ROOT_DIR+r'/output/prop_test.csv')

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

'''
dl[(dl['Sex']=='All')&(dl['Race']=='All')&(dl['Type']=='Naive')&(dl['State']==0)]
prop[(prop['Sex']=='All')&(prop['Type']=='Naive') & (prop['State']==0)]
'''

#################
#FIXES
##################
dat = dt.loc[(dt['Age']=='All') & (dt['Sex']=='1') & (dt['Race']!='All'),'Population']
for ty in pd.unique(dl['Type']):
    for st in pd.unique(dl['State']):
        for r in pd.unique(dl['Race']):
            if r !='All': dl.loc[(dl['Type']==ty)&(dl['State']==st) & (dl['Race']==r) & (dl['Sex']=='1') & (dl['Age']=='All'),'Sex'] = ['All','1']
