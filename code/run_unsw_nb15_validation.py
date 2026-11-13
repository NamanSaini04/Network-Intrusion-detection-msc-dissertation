"""UNSW-NB15 cross-dataset benchmark experiment matching the revised dissertation methodology.

Uses a proportional stratified training down-sample of 60,000 records for computational feasibility,
retains the full official 82,332-record test set, fits preprocessing on training data only,
and trains MLP/CNN/LSTM for the documented one epoch.
"""
from __future__ import annotations
import argparse, json, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
SEED=42; K_BEST=40; EPOCHS=1; BATCH_SIZE=8192
TRAIN_TARGETS={'Normal':19162,'Generic':13687,'Exploits':11426,'Fuzzers':6222,'DoS':4196,'Reconnaissance':3589,'Analysis':684,'Backdoor':597,'Shellcode':387,'Worms':50}

def set_seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def sample_exact(train):
    parts=[]
    for cls,n in TRAIN_TARGETS.items():
        g=train[train['attack_cat'].eq(cls)]
        if len(g)<n: raise ValueError(f'Not enough rows for {cls}: need {n}, found {len(g)}')
        parts.append(g.sample(n=n,random_state=SEED))
    return pd.concat(parts,ignore_index=True).sample(frac=1,random_state=SEED).reset_index(drop=True)

def eval_pred(name,y_test,pred,labels,out):
    macro=precision_recall_fscore_support(y_test,pred,average='macro',zero_division=0); weighted=precision_recall_fscore_support(y_test,pred,average='weighted',zero_division=0)
    row={'model':name,'accuracy':accuracy_score(y_test,pred),'macro_precision':macro[0],'macro_recall':macro[1],'macro_f1':macro[2],'weighted_f1':weighted[2]}
    pd.DataFrame(classification_report(y_test,pred,target_names=labels,output_dict=True,zero_division=0)).T.reset_index().rename(columns={'index':'class'}).to_csv(out/'tables'/f'unsw_{name}_classification_report.csv',index=False)
    cm=confusion_matrix(y_test,pred); pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/'tables'/f'unsw_{name}_confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(9,8)); ConfusionMatrixDisplay(cm,display_labels=labels).plot(include_values=False,xticks_rotation=90,ax=ax,colorbar=True); ax.set_title(f'UNSW-NB15 - {name}'); fig.tight_layout(); fig.savefig(out/'figures'/f'unsw_{name}_confusion_matrix.png',dpi=200); plt.close(fig)
    return row
class MLP(nn.Module):
    def __init__(self,nf,nc): super().__init__(); self.net=nn.Sequential(nn.Linear(nf,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,nc))
    def forward(self,x): return self.net(x)
class CNN1D(nn.Module):
    def __init__(self,nf,nc): super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveMaxPool1d(1),nn.Flatten(),nn.Linear(32,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,nc))
    def forward(self,x): return self.net(x.unsqueeze(1))
class LSTMFeat(nn.Module):
    def __init__(self,nf,nc): super().__init__(); self.lstm=nn.LSTM(1,32,batch_first=True); self.fc=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,nc))
    def forward(self,x): out,_=self.lstm(x.unsqueeze(-1)); return self.fc(out[:,-1,:])

def train_torch(name,model,Xtr,ytr,Xte,yte,labels,weights,out):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr.astype('int64'))),batch_size=BATCH_SIZE,shuffle=True)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device)); opt=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(EPOCHS):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(Xte),BATCH_SIZE): preds.append(model(torch.from_numpy(Xte[i:i+BATCH_SIZE]).to(device)).argmax(1).cpu().numpy())
    return eval_pred(name,yte,np.concatenate(preds),labels,out)

def save_unsw_test_distribution_figure(test, out):
    """Generate Figure 5.3 from the untouched official test distribution only."""
    order = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms']
    counts = test['attack_cat'].value_counts().reindex(order).fillna(0).astype(int)
    percentages = counts / counts.sum() * 100
    dist = pd.DataFrame({'class': counts.index, 'count': counts.values, 'percentage': percentages.values})
    dist.to_csv(out/'tables'/'unsw_table_5_2_official_test_class_distribution.csv', index=False)
    plot_df = dist.sort_values('count', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot_df['class'], plot_df['count'])
    ax.set_xlabel('Number of records')
    ax.set_title('UNSW-NB15 Official Test Class Distribution')
    ax.grid(axis='x', alpha=0.25)
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(row.count + max(dist['count']) * 0.01, i, f'{row.count:,} ({row.percentage:.1f}%)', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'figures'/'figure_5_3_unsw_nb15_official_test_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_unsw_model_comparison_figure(rows, out):
    """Generate Figure 5.4 using the UNSW-NB15 macro F1 results."""
    df = pd.DataFrame(rows).sort_values('macro_f1', ascending=True)
    labels = [str(x).replace('_',' ').title().replace('Sgd','SGD').replace('Mlp','MLP').replace('Cnn','CNN').replace('Lstm','LSTM') for x in df['model']]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, df['macro_f1'])
    ax.set_xlabel('Macro F1-score')
    ax.set_title('UNSW-NB15 Model Comparison by Macro F1-score')
    ax.grid(axis='x', alpha=0.25)
    for i, value in enumerate(df['macro_f1']):
        ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'figures'/'figure_5_4_unsw_model_comparison_macro_f1.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-file',default='data/unsw_nb15/UNSW_NB15_training-set.csv'); ap.add_argument('--test-file',default='data/unsw_nb15/UNSW_NB15_testing-set.csv'); ap.add_argument('--output-dir',default='analysis_outputs/unsw_nb15'); args=ap.parse_args()
    set_seed(); out=Path(args.output_dir); (out/'tables').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(args.train_file); test=pd.read_csv(args.test_file)
    for df in (train,test):
        df.columns=[c.strip().replace('\ufeff','') for c in df.columns]; df['attack_cat']=df['attack_cat'].astype(str).str.strip().str.replace('Backdoors','Backdoor',regex=False)
    train=sample_exact(train)
    save_unsw_test_distribution_figure(test, out)
    pd.DataFrame([(c,TRAIN_TARGETS[c],int((test.attack_cat==c).sum())) for c in TRAIN_TARGETS],columns=['class','train_count','test_count']).to_csv(out/'tables'/'unsw_train_test_class_distribution.csv',index=False)
    le=LabelEncoder(); ytr=le.fit_transform(train.attack_cat); yte=le.transform(test.attack_cat); labels=list(le.classes_); nc=len(labels)
    Xtr_df=train.drop(columns=['id','label','attack_cat'],errors='ignore'); Xte_df=test.drop(columns=['id','label','attack_cat'],errors='ignore')
    cat=[c for c in Xtr_df if Xtr_df[c].dtype=='object']; num=[c for c in Xtr_df if c not in cat]
    try: ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError: ohe=OneHotEncoder(handle_unknown='ignore',sparse=False)
    pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ohe',ohe)]),cat)],sparse_threshold=0.0)
    Xtr_all=pre.fit_transform(Xtr_df); Xte_all=pre.transform(Xte_df); sel=SelectKBest(f_classif,k=min(K_BEST,Xtr_all.shape[1])); Xtr=sel.fit_transform(Xtr_all,ytr).astype('float32'); Xte=sel.transform(Xte_all).astype('float32')
    names=pre.get_feature_names_out(); pd.DataFrame({'selected_feature':names[sel.get_support()]}).to_csv(out/'tables'/'unsw_selected_features.csv',index=False)
    weights=compute_class_weight('balanced',classes=np.arange(nc),y=ytr); rows=[]
    models={
      'sgd_logistic':SGDClassifier(loss='log_loss',class_weight='balanced',max_iter=1500,tol=1e-3,random_state=SEED),
      'sgd_linear_svm':SGDClassifier(loss='hinge',class_weight='balanced',max_iter=1500,tol=1e-3,random_state=SEED),
      'random_forest':RandomForestClassifier(n_estimators=80,max_depth=16,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),
      'xgboost':XGBClassifier(n_estimators=40,max_depth=4,learning_rate=.08,subsample=.9,colsample_bytree=.9,objective='multi:softmax',num_class=nc,eval_metric='mlogloss',random_state=SEED,tree_method='hist')
    }
    for name,m in models.items(): m.fit(Xtr,ytr); rows.append(eval_pred(name,yte,m.predict(Xte),labels,out))
    rows.append(train_torch('mlp_neural_network',MLP(40,nc),Xtr,ytr,Xte,yte,labels,weights,out)); rows.append(train_torch('cnn_1d',CNN1D(40,nc),Xtr,ytr,Xte,yte,labels,weights,out)); rows.append(train_torch('lstm_features',LSTMFeat(40,nc),Xtr,ytr,Xte,yte,labels,weights,out))
    pd.DataFrame(rows).sort_values('macro_f1',ascending=False).to_csv(out/'tables'/'unsw_model_metrics_summary.csv',index=False)
    save_unsw_model_comparison_figure(rows, out)
    (out/'run_evidence').mkdir(parents=True, exist_ok=True)
    metadata={'seed':SEED,'train_rows':len(train),'test_rows':len(test),'selected_features':40,'epochs':EPOCHS,'batch_size':BATCH_SIZE,'official_test_retained':True,'sampling':'proportional stratified training down-sample'}
    json.dump(metadata,open(out/'tables'/'run_metadata.json','w'),indent=2)
    json.dump(metadata,open(out/'run_evidence'/'unsw_run_metadata.json','w'),indent=2)
if __name__=='__main__': main()
