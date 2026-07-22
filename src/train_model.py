"""Leakage-aware chronological experiments for HomeC anomaly detection.

Outputs reproducible baselines, global-vs-local label diagnostics, model artifacts,
and a machine-readable experiment summary. The deployable model excludes
``total_appliance`` because it is an additive component of ``use [kW]``.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

BASE_DIR=Path(__file__).resolve().parents[1]
DATA_PATH=BASE_DIR/'data'/'HomeC_cleaned_final.zip'
OUT_DIR=BASE_DIR/'notebooks'; OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE=42
LEAKY_SOURCE=["gen [kW]","total_appliance","temperature","humidity","hour","dayofweek","month","is_weekend"]
CLEAN_SOURCE=[c for c in LEAKY_SOURCE if c!="total_appliance"]
RENAME={"gen [kW]":"gen_kw"}


def choose_threshold(y, score):
    p,r,t=precision_recall_curve(y,score)
    if len(t)==0: return 0.5,0.0
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    i=int(np.nanargmax(f)); return float(t[i]),float(f[i])

def metrics(y,score,thr):
    pred=(score>=thr).astype('int8')
    auc=float(roc_auc_score(y,score)) if len(np.unique(y))>1 else float('nan')
    return {"F1":round(float(f1_score(y,pred,zero_division=0)),4),"AUC":round(auc,4),
            "Precision":round(float(precision_score(y,pred,zero_division=0)),4),
            "Recall":round(float(recall_score(y,pred,zero_division=0)),4)},pred

def score_model(m,X):
    if hasattr(m,'predict_proba'): return m.predict_proba(X)[:,1]
    # IsolationForest uses lower decision scores for more abnormal points.
    if isinstance(m,IsolationForest): return -m.decision_function(X)
    if hasattr(m,'decision_function'): return m.decision_function(X)
    raise TypeError('Model has no score method')

def monthly_rates(dt,y):
    z=pd.DataFrame({'month':dt.dt.month,'y':y})
    return {str(int(k)):round(float(v)*100,3) for k,v in z.groupby('month')['y'].mean().items()}

def make_labels(df,train_end):
    use=df['use [kW]'].astype(float)
    train=use.iloc[:train_end]
    global_thr=float(train.mean()+3*train.std())
    global_y=(use>global_thr).astype('int8')
    # Prior 30-day context only: shift prevents the current value entering its own threshold.
    prior=use.shift(1)
    roll_mean=prior.rolling('30D',min_periods=1440).mean()
    roll_std=prior.rolling('30D',min_periods=1440).std()
    fallback_mean=float(train.mean()); fallback_std=float(train.std())
    local_thr=(roll_mean.fillna(fallback_mean)+3*roll_std.fillna(fallback_std))
    local_y=(use>local_thr).astype('int8')
    return global_y,local_y,global_thr,local_thr

def baseline_total_appliance(Xv,yv,Xt,yt):
    thr,_=choose_threshold(yv,Xv['total_appliance'].to_numpy())
    m,_=metrics(yt,Xt['total_appliance'].to_numpy(),thr)
    return {"Experiment":"Baseline: total_appliance threshold","Features":"total_appliance only","Label":"train-global 3 sigma",**m,"DecisionThreshold":round(thr,6)}

def fit_experiment(name,features,y,df,train_end,val_end,models,fit_step=10):
    X=df[features].rename(columns=RENAME).astype('float32')
    Xt,Xv,Xs=X.iloc[:train_end],X.iloc[train_end:val_end],X.iloc[val_end:]
    yt,yv,ys=y.iloc[:train_end],y.iloc[train_end:val_end],y.iloc[val_end:]
    Xfit,yfit=Xt.iloc[::fit_step],yt.iloc[::fit_step]
    rows=[]; fitted={}; thresholds={}
    for model_name,model in models.items():
        model.fit(Xfit,yfit); sv=score_model(model,Xv); thr,_=choose_threshold(yv,sv)
        vm,_=metrics(yv,sv,thr)
        final=clone(model); final.fit(pd.concat([Xt,Xv]).iloc[::5],pd.concat([yt,yv]).iloc[::5])
        ss=score_model(final,Xs); tm,pred=metrics(ys,ss,thr)
        rows.append({"Experiment":name,"Model":model_name,"Features":len(features),"ValidationF1":vm['F1'],**{f"Test{k}":v for k,v in tm.items()},"DecisionThreshold":round(thr,6)})
        fitted[model_name]=(final,thr,tm,confusion_matrix(ys,pred).tolist())
    return rows,fitted

def main():
    df=pd.read_csv(DATA_PATH,low_memory=False)
    df['datetime']=pd.to_datetime(df['datetime']); df=df.sort_values('datetime').reset_index(drop=True)
    # DatetimeIndex enables time-based rolling windows.
    indexed=df.set_index('datetime',drop=False)
    n=len(df); train_end=int(n*.70); val_end=int(n*.85)
    global_y,local_y,global_thr,local_thr=make_labels(indexed,train_end)
    global_y=global_y.reset_index(drop=True); local_y=local_y.reset_index(drop=True)
    Xleaky=df[LEAKY_SOURCE].rename(columns=RENAME).astype('float32')

    rows=[]
    # Required non-ML baselines.
    majority_score=np.zeros(len(df)-val_end)
    mm,_=metrics(global_y.iloc[val_end:],majority_score,0.5)
    rows.append({"Experiment":"Baseline: majority normal","Features":"none","Label":"train-global 3 sigma",**mm,"DecisionThreshold":0.5})
    rows.append(baseline_total_appliance(Xleaky.iloc[train_end:val_end],global_y.iloc[train_end:val_end],Xleaky.iloc[val_end:],global_y.iloc[val_end:]))

    pos=float(global_y.iloc[:train_end:3].sum()); neg=len(global_y.iloc[:train_end:3])-pos
    pw=neg/max(pos,1)
    compact={
      'LogisticRegression':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=500,class_weight='balanced',random_state=42))]),
      'LightGBM':LGBMClassifier(n_estimators=45,num_leaves=31,learning_rate=.08,class_weight='balanced',n_jobs=4,random_state=42,verbose=-1),
    }
    r1,_=fit_experiment('Global label + leaky features',LEAKY_SOURCE,global_y,df,train_end,val_end,compact)
    r2,_=fit_experiment('Global label + leakage removed',CLEAN_SOURCE,global_y,df,train_end,val_end,compact)
    rows.extend(r1+r2)

    pos=float(local_y.iloc[:train_end:3].sum()); neg=len(local_y.iloc[:train_end:3])-pos; pw=neg/max(pos,1)
    contamination=max(min(pos/max(pos+neg,1),.1),.001)
    full={
      'LogisticRegression':compact['LogisticRegression'],
      'RandomForest':RandomForestClassifier(n_estimators=30,max_depth=14,class_weight='balanced_subsample',n_jobs=-1,random_state=42),
      'XGBoost':XGBClassifier(n_estimators=40,max_depth=4,learning_rate=.08,subsample=.85,colsample_bytree=.9,scale_pos_weight=pw,tree_method='hist',eval_metric='logloss',n_jobs=4,random_state=42),
      'LightGBM':compact['LightGBM'],
      'IsolationForest':IsolationForest(n_estimators=80,contamination=contamination,n_jobs=-1,random_state=42),
    }
    r3,fitted=fit_experiment('Primary: local 30-day label + leakage removed',CLEAN_SOURCE,local_y,df,train_end,val_end,full)
    rows.extend(r3)
    results=pd.DataFrame(rows)
    results.to_csv(OUT_DIR/'experiment_results.csv',index=False)
    # Keep model comparison compatible with dashboard/repository.
    primary=results[results['Experiment'].eq('Primary: local 30-day label + leakage removed')].copy()
    comparison=primary[['Model','TestF1','TestAUC','TestPrecision','TestRecall','DecisionThreshold']].rename(
      columns={'TestF1':'F1','TestAUC':'AUC','TestPrecision':'Precision','TestRecall':'Recall'})
    comparison[['Model','F1','AUC','Precision','Recall','DecisionThreshold']].to_csv(OUT_DIR/'model_comparison.csv',index=False)
    best_row=primary.sort_values('ValidationF1',ascending=False).iloc[0]; best=str(best_row['Model'])
    model,thr,testm,cm=fitted[best]
    model_features=[RENAME.get(c,c) for c in CLEAN_SOURCE]
    joblib.dump(model,OUT_DIR/'best_model.pkl'); joblib.dump(model_features,OUT_DIR/'feature_columns.pkl')
    Xclean=df[CLEAN_SOURCE].rename(columns=RENAME)
    defaults={c:float(Xclean.iloc[:val_end][c].median()) for c in model_features}
    (OUT_DIR/'feature_defaults.json').write_text(json.dumps(defaults,indent=2),encoding='utf-8')
    split_rates={
      'global':{'train':round(float(global_y.iloc[:train_end].mean())*100,3),'validation':round(float(global_y.iloc[train_end:val_end].mean())*100,3),'test':round(float(global_y.iloc[val_end:].mean())*100,3)},
      'local_30d':{'train':round(float(local_y.iloc[:train_end].mean())*100,3),'validation':round(float(local_y.iloc[train_end:val_end].mean())*100,3),'test':round(float(local_y.iloc[val_end:].mean())*100,3)}}
    metadata={
      'selected_model':best,'decision_threshold':round(float(thr),8),'test_metrics':testm,'confusion_matrix':cm,
      'split_strategy':'chronological 70/15/15','label_definition':'use [kW] exceeds prior 30-day rolling mean + 3 rolling std; current row excluded',
      'global_threshold_train_kw':round(global_thr,6),'source_features':CLEAN_SOURCE,'model_features':model_features,
      'leakage_control':'total_appliance excluded from deployable model because it is an additive component of use [kW]',
      'split_anomaly_rates_percent':split_rates,
      'monthly_global_rates_percent':monthly_rates(df['datetime'],global_y),
      'monthly_local_rates_percent':monthly_rates(df['datetime'],local_y),
      'limitations':['Proxy high-load label is not a verified fault label.','One household limits external validity.','Rolling threshold adapts to seasonality but may not absorb long regime shifts immediately.','Timeline relies on one-reading-per-minute reconstruction.']}
    (OUT_DIR/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(results.to_string(index=False)); print('\nSelected',best,testm)
if __name__=='__main__': main()
