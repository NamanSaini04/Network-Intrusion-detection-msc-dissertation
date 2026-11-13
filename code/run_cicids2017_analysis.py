"""CICIDS2017 primary benchmark analysis matching the revised dissertation methodology.

Place raw CICIDS2017 CSV files under data/cicids2017/raw/ or pass --raw-dir.
Sampling is performed before feature scaling and ANOVA F-score selection. CNN and
LSTM use separate documented training histories: CNN=80 epochs, LSTM=15 epochs,
with batch size 1024. These neural runs are indicative pipeline experiments, not
optimised deep-learning benchmarks.
"""
from __future__ import annotations
import argparse, glob, json, random, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

SEED = 42
K_BEST = 40
CNN_EPOCHS = 80
LSTM_EPOCHS = 15
NEURAL_BATCH_SIZE = 1024
XGB_TRAIN_ROWS = 6068
DROP_COLS = ['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp','SimillarHTTP','Unnamed: 0','Inbound']
CAPS = {
    'BENIGN':4000, 'PortScan':3500, 'DDoS':3500, 'DoS Hulk':3500,
    'FTP-Patator':2500, 'DoS Slowhttptest':2500, 'DoS GoldenEye':2500,
    'SSH-Patator':2500, 'DoS slowloris':2500
}

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalise_label(s):
    s = str(s).strip().replace('�','-')
    s = ' '.join(s.split())
    s = s.replace('Web Attack - Sql Injection','Web Attack - SQL Injection')
    return s

def find_csvs(raw_dir):
    files = sorted(glob.glob(str(Path(raw_dir)/'**'/'*.csv'), recursive=True))
    if not files: raise FileNotFoundError(f'No CICIDS2017 CSV files found under {raw_dir}')
    return files

def count_labels(raw_dir):
    counts=Counter()
    for f in find_csvs(raw_dir):
        for chunk in pd.read_csv(f, chunksize=300000, low_memory=False):
            chunk.columns=[c.strip() for c in chunk.columns]
            if 'Label' in chunk:
                counts.update(chunk['Label'].map(normalise_label).value_counts().to_dict())
    return counts

def build_sample(raw_dir, counts, seed=SEED):
    rng=np.random.default_rng(seed)
    targets={lab:min(CAPS.get(lab,cnt),cnt) for lab,cnt in counts.items()}
    remain=targets.copy(); parts=[]
    for f in find_csvs(raw_dir):
        df=pd.read_csv(f, low_memory=False); df.columns=[c.strip() for c in df.columns]
        if 'Label' not in df: continue
        df['Label']=df['Label'].map(normalise_label)
        for lab,need in list(remain.items()):
            if need<=0: continue
            g=df[df['Label'].eq(lab)]
            if g.empty: continue
            take=min(need,len(g))
            part=g.sample(n=take, random_state=int(rng.integers(0,1_000_000_000))) if take<len(g) else g
            parts.append(part); remain[lab]-=take
    return pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)

def clean(df):
    y=df['Label'].map(normalise_label)
    X=df.drop(columns=['Label'],errors='ignore').drop(columns=[c for c in DROP_COLS if c in df.columns],errors='ignore')
    X=X.loc[:,~X.columns.duplicated()]
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors='coerce')
    X=X.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how='all')
    # This median imputation occurs during cleaning in the supplied implementation and is documented as a limited leakage risk.
    X=X.fillna(X.median(numeric_only=True)).fillna(0)
    return X.select_dtypes(include=[np.number]), y

def preprocess(Xtr_df,Xte_df,ytr,k=K_BEST):
    vt=VarianceThreshold(0.0); Xtr_v=vt.fit_transform(Xtr_df); Xte_v=vt.transform(Xte_df)
    names=list(Xtr_df.columns[vt.get_support()])
    sel=SelectKBest(f_classif,k=min(k,Xtr_v.shape[1])); Xtr_s=sel.fit_transform(Xtr_v,ytr); Xte_s=sel.transform(Xte_v)
    selected=[n for n,keep in zip(names,sel.get_support()) if keep]
    sc=StandardScaler(); return sc.fit_transform(Xtr_s).astype('float32'), sc.transform(Xte_s).astype('float32'), selected, vt, sel, sc

def metrics_row(name,y,pred):
    macro=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    weighted=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    return {'model':name,'accuracy':accuracy_score(y,pred),'macro_precision':macro[0],'macro_recall':macro[1],'macro_f1':macro[2],'weighted_f1':weighted[2]}

def evaluate(name,y,pred,labels,out):
    pd.DataFrame(classification_report(y,pred,target_names=labels,output_dict=True,zero_division=0)).T.reset_index().rename(columns={'index':'class'}).to_csv(out/'tables'/f'{name}_classification_report.csv',index=False)
    cm=confusion_matrix(y,pred); pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/'tables'/f'{name}_confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(10,9)); ConfusionMatrixDisplay(cm,display_labels=labels).plot(include_values=False,xticks_rotation=90,ax=ax,colorbar=True); ax.set_title(f'CICIDS2017 - {name}'); fig.tight_layout(); fig.savefig(out/'figures'/f'{name}_confusion_matrix.png',dpi=200); plt.close(fig)
    return metrics_row(name,y,pred)

def stratified_limit(X, y, n_rows=XGB_TRAIN_ROWS, seed=SEED):
    if len(y) <= n_rows:
        return X, y
    rng = np.random.default_rng(seed)
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    allocated = {cls: max(1, int(round(n_rows * cnt / len(y)))) for cls, cnt in zip(classes, counts)}
    diff = n_rows - sum(allocated.values())
    order = list(classes[np.argsort(-counts)])
    idx = 0
    while diff != 0:
        cls = order[idx % len(order)]
        if diff > 0:
            allocated[cls] += 1; diff -= 1
        elif allocated[cls] > 1:
            allocated[cls] -= 1; diff += 1
        idx += 1
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        take = min(len(cls_idx), allocated[cls])
        selected.extend(rng.choice(cls_idx, size=take, replace=False))
    selected = np.array(selected)
    rng.shuffle(selected)
    return X[selected], y[selected]

class CNN1D(nn.Module):
    def __init__(self,nf,nc):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveMaxPool1d(1),nn.Flatten(),nn.Linear(32,64),nn.ReLU(),nn.Dropout(.20),nn.Linear(64,nc))
    def forward(self,x): return self.net(x.unsqueeze(1))
class LSTMFeat(nn.Module):
    def __init__(self,nf,nc):
        super().__init__(); self.lstm=nn.LSTM(1,32,batch_first=True); self.fc=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Dropout(.20),nn.Linear(64,nc))
    def forward(self,x): out,_=self.lstm(x.unsqueeze(-1)); return self.fc(out[:,-1,:])

def train_torch(name,model,Xtr,ytr,Xte,yte,labels,weights,out,epochs,batch_size):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr.astype('int64'))),batch_size=batch_size,shuffle=True)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device)); opt=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(epochs):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(Xte),batch_size): preds.append(model(torch.from_numpy(Xte[i:i+batch_size]).to(device)).argmax(1).cpu().numpy())
    return evaluate(name,yte,np.concatenate(preds),labels,out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',default='data/cicids2017/raw')
    ap.add_argument('--output-dir',default='analysis_outputs/cicids2017_reproduced')
    ap.add_argument('--cnn-epochs',type=int,default=CNN_EPOCHS)
    ap.add_argument('--lstm-epochs',type=int,default=LSTM_EPOCHS)
    ap.add_argument('--neural-batch-size',type=int,default=NEURAL_BATCH_SIZE)
    ap.add_argument('--xgb-train-rows',type=int,default=XGB_TRAIN_ROWS)
    args=ap.parse_args()
    set_seed(); out=Path(args.output_dir); (out/'tables').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True); (out/'run_evidence').mkdir(parents=True,exist_ok=True)
    counts=count_labels(args.raw_dir); sample=build_sample(args.raw_dir,counts)
    X,yraw=clean(sample); le=LabelEncoder(); y=le.fit_transform(yraw); labels=list(le.classes_)
    Xtr_df,Xte_df,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    dist=[]
    for i,label in enumerate(labels):
        total=int((y==i).sum()); tr=int((ytr==i).sum()); te=int((yte==i).sum()); dist.append((label,total,tr,te))
    pd.DataFrame(dist,columns=['class','sample_count','train_count','test_count']).to_csv(out/'tables'/'cicids_train_test_class_distribution.csv',index=False)
    Xtr,Xte,selected,vt,sel,sc=preprocess(Xtr_df,Xte_df,ytr); pd.DataFrame({'selected_feature':selected}).to_csv(out/'tables'/'selected_features.csv',index=False)
    class_weights=compute_class_weight('balanced',classes=np.arange(len(labels)),y=ytr)
    rows=[]
    models={
      'sgd_logistic':SGDClassifier(loss='log_loss',max_iter=2000,tol=1e-3,random_state=SEED,class_weight='balanced'),
      'sgd_linear_svm':SGDClassifier(loss='hinge',max_iter=2000,tol=1e-3,random_state=SEED,class_weight='balanced'),
      'random_forest':RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),
    }
    for name,m in models.items():
        m.fit(Xtr,ytr); rows.append(evaluate(name,yte,m.predict(Xte),labels,out))
    # CICIDS2017 XGBoost is fitted on a documented 6,068-row stratified training subset for runtime consistency with reported output metadata.
    Xgb_Xtr, Xgb_ytr = stratified_limit(Xtr, ytr, n_rows=args.xgb_train_rows, seed=SEED)
    xgb_weights = compute_class_weight('balanced', classes=np.arange(len(labels)), y=Xgb_ytr)
    xgb_sample_weight = np.array([xgb_weights[c] for c in Xgb_ytr])
    xgb = XGBClassifier(n_estimators=70,max_depth=4,learning_rate=.08,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=len(labels),eval_metric='mlogloss',random_state=SEED,tree_method='hist')
    xgb.fit(Xgb_Xtr, Xgb_ytr, sample_weight=xgb_sample_weight)
    rows.append(evaluate('xgboost_reduced_subset',yte,xgb.predict(Xte),labels,out))
    mlp=MLPClassifier(hidden_layer_sizes=(96,48),activation='relu',solver='adam',learning_rate_init=.001,alpha=1e-4,max_iter=80,early_stopping=True,random_state=SEED)
    try:
        mlp_weights = np.array([compute_class_weight('balanced',classes=np.arange(len(labels)),y=ytr)[c] for c in ytr])
        mlp.fit(Xtr,ytr,sample_weight=mlp_weights)
    except TypeError:
        mlp.fit(Xtr,ytr)
    rows.append(evaluate('mlp_neural_network',yte,mlp.predict(Xte),labels,out))
    rows.append(train_torch('cnn_1d',CNN1D(40,len(labels)),Xtr,ytr,Xte,yte,labels,class_weights,out,args.cnn_epochs,args.neural_batch_size))
    rows.append(train_torch('lstm_features',LSTMFeat(40,len(labels)),Xtr,ytr,Xte,yte,labels,class_weights,out,args.lstm_epochs,args.neural_batch_size))
    pd.DataFrame(rows).sort_values('macro_f1',ascending=False).to_csv(out/'tables'/'model_metrics_summary.csv',index=False)
    metadata={
        'seed':SEED,
        'sample_rows':len(sample),
        'train_rows':len(ytr),
        'test_rows':len(yte),
        'selected_features':40,
        'xgboost_training_rows':len(Xgb_ytr),
        'cnn_epochs':args.cnn_epochs,
        'lstm_epochs':args.lstm_epochs,
        'neural_batch_size':args.neural_batch_size,
        'models_completed':['sgd_logistic','sgd_linear_svm','random_forest','xgboost_reduced_subset','mlp_neural_network','cnn_1d','lstm_features']
    }
    json.dump(metadata,open(out/'tables'/'run_metadata.json','w'),indent=2)
    json.dump(metadata,open(out/'run_evidence'/'main_run_metadata.json','w'),indent=2)
if __name__=='__main__': main()
