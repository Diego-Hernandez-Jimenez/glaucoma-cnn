

# utility functions for master's thesis

from time import time
from os import listdir
from copy import deepcopy
from PIL import Image
from cv2 import createCLAHE,cvtColor,COLOR_RGB2LAB,COLOR_LAB2RGB
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

import torch
import torchvision.transforms as transf
from torch.utils.data import Dataset
import torch.nn as nn




## LOADING DATA

def extract_labels(img_list:list):
  return [1. if '_g_' in img else 0. for img in img_list]



class CLAHE_transf():
  def __init__(self,cliplimit=5,tilegridsize=(5,5)):
    self.cliplim = cliplimit
    self.tilesize = tilegridsize
  
  def __call__(self,im_pil):
    # cv2 requires numpy objects
    im_np = transf.functional.pil_to_tensor(im_pil).numpy()
    clahe = createCLAHE(clipLimit=self.cliplim,tileGridSize=self.tilesize)
    # rgb -> l*a*b (opencv assumes shape of [Height,Width,Channel], hence the transpose)
    im_lab = cvtColor(im_np.transpose(1,2,0),COLOR_RGB2LAB)
    # apply clahe to l channel
    im_lab[:,:,0] = clahe.apply(im_lab[:,:,0])
    # l*a*b -> rgb
    im_rgb = cvtColor(im_lab,COLOR_LAB2RGB)
    return im_rgb

class ImgData(Dataset):
  
  def __init__(self,img_list,labels,path,transform):
    self.img_location = path
    self.imgs = img_list
    self.labels = labels
    self.transform = transform

  def __getitem__(self,idx): 
    img = Image.open(self.img_location + self.imgs[idx])
    label = self.labels[idx]
    img = self.transform(img)

    return img,label

  def item_name(self,idx):
    return self.imgs[idx]

  def __len__(self):
    return len(self.labels)




## TRAINING

def mixup_fn(X,y,alpha=0.1,device=None):
    """
    Inspired by
    https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py

    """
    lambd = np.random.beta(alpha,alpha)
    batch_size = X.size(0)
    ids = torch.randperm(batch_size,device=device)

    mixed_X = lambd*X + (1-lambd)*X[ids,:]
    mixed_y = lambd*y + (1-lambd)*y[ids,:]
    
    return mixed_X,mixed_y

def train_model(
  model:nn.Module,
  epochs:int,
  train_dl,
  val_dl,
  cost:nn.Module,
  optimizer:nn.Module,
  alpha_mixup:float = 0.,
  smooth:float = 0.,
  scheduler = None,
  ES:dict = None,
  device:str = 'cuda'
) -> tuple:
  """
  Generic function for NN training. Inspired by Raschka, Liu & Mirjalili (2022)
  and Pytorch documentation

  """
  start = time()
  train_loss_hist = [0]*epochs
  train_acc_hist = [0]*epochs
  val_loss_hist = [0]*epochs
  val_acc_hist = [0]*epochs
  p = 0 # patience counter for early stopping
  prev_val_acc = torch.tensor(0,device=device) # initial validation acc, needed to perform early stopping

  # updated if early stopping is activated
  best_params = deepcopy(model.state_dict())
  
  for epoch in range(epochs):

    # print(scheduler.get_last_lr())
    print(f'Epoch {epoch+1}/{epochs}')

    model.train() # training mode
    loss_sum = 0.
    acc_sum = 0
    for X,y in train_dl:
      X,y = X.to(device), y.to(device).unsqueeze(1) # shape X: [Batch,Channel,Height,Width] || shape y: [Batch,1]
      if alpha_mixup > 0:
        X,y = mixup_fn(X,y,alpha_mixup,device)
      output = model(X) # forward propagation
      optimizer.zero_grad() # reset gradients to 0
      loss = cost(output,y*(1-smooth)+smooth/2)
      loss.backward() # gradient computation (backprop)
      optimizer.step() # params update 

      loss_sum += loss.item()
      preds = torch.sigmoid(output) > 0.5
      is_correct = (preds == y)
      acc_sum += is_correct.sum() 

    train_loss_hist[epoch] = loss_sum / len(train_dl)
    train_acc_hist[epoch] = acc_sum.double() / len(train_dl.dataset)

    model.eval() # evaluation mode
    loss_sum = 0.
    acc_sum = 0
    with torch.no_grad():
      for X,y in val_dl:
        X,y = X.to(device), y.to(device).unsqueeze(1)
        output = model(X)
        
        loss = cost(output,y*(1-smooth)+smooth/2)

        loss_sum += loss.item()
        preds = torch.sigmoid(output) > 0.5
        is_correct = (preds == y)
        acc_sum += is_correct.sum() 
        
      val_loss_hist[epoch] = loss_sum / len(val_dl)
      val_acc_hist[epoch] = acc_sum.double() / len(val_dl.dataset)
      # learning rate update 
      if scheduler is not None:
          scheduler.step()

    print(f"""
                   | Train     | Validation |
    |--------------|-----------|------------|
    | Loss         | {train_loss_hist[epoch]:.4f}    | {val_loss_hist[epoch]:.4f}     |
    | Accuracy     | {train_acc_hist[epoch]:.4f}    | {val_acc_hist[epoch]:.4f}     |
    """)
    
  # early stopping
  
    if ES is not None:
      if (val_acc_hist[epoch] - prev_val_acc) > ES['delta']: # if current > previous by some delta margin, then update best params
        best_params = deepcopy(model.state_dict())
        print(f'best parameteres are from epoch {epoch+1}')
        prev_val_acc = val_acc_hist[epoch]
        p = 0
        continue
      else:
        p += 1
        if p == ES['patience']:
          elapsed = time() - start
          print(f'Completed in {int(elapsed//60)} min {elapsed%60:.0f} s')
          model.load_state_dict(best_params)

          return (train_loss_hist[:epoch+1],train_acc_hist[:epoch+1]),\
                 (val_loss_hist[:epoch+1],val_acc_hist[:epoch+1])


  elapsed = time() - start
  print(f'Completed in {int(elapsed//60)} min {elapsed%60:.0f} s')

  return (train_loss_hist,train_acc_hist),(val_loss_hist,val_acc_hist)




## LEARNING CURVES

def plot_learning_curves(train_hist:tuple,val_hist:tuple,savedata:str = None,saveplot:str = None):
  loss_train = train_hist[0]
  accs_train = [float(x.cpu()) for x in train_hist[1]]
  loss_val = val_hist[0]
  accs_val = [float(x.cpu()) for x in val_hist[1]]
  # In case we want to save the results for future use
  if savedata is not None:
    from pandas import DataFrame
    d = {'train_loss':loss_train,'train_acc':accs_train,'val_loss':loss_val,'val_acc':accs_val}
    DataFrame(d).to_csv(f'/content/drive/MyDrive/Colab_Notebooks/GLAUCOMA/plots/train_val_history_{savedata}.txt')

  with plt.style.context('seaborn-talk'):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8*2,5))

    ax1.plot(loss_train,label='train')
    ax1.plot(loss_val,label='validation')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend(loc='upper right')

    ax2.plot(accs_train,label='train')
    ax2.plot(accs_val,label='validation')
    # ax2.set_ylim([0,1])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend(loc='lower right')

    if saveplot is not None:
      plt.savefig(f'/content/drive/MyDrive/Colab_Notebooks/GLAUCOMA/plots/loss_acc{saveplot}.jpg',dpi=300)

## PERFORMANCE

def predict(model:nn.Module,data:ImgData,device:str):
  outputs = torch.zeros(len(data))
  model.eval()
  with torch.no_grad():
    for i in range(len(data)):
      outputs[i] = model(data[i][0].unsqueeze(0).to(device))
    
    probs = torch.sigmoid(outputs)
    preds = probs > 0.5
    
    
  return preds,probs

def confmat(test_labels,preds,as_prob=None):
  fig,ax = plt.subplots(figsize=(6,6))

  metrics.ConfusionMatrixDisplay.from_predictions(test_labels,preds,
                                                  labels=[0,1],
                                                  normalize=as_prob,
                                                  display_labels=['no glaucoma','glaucoma'],
                                                  cmap='Blues', # Blues RdYlGn
                                                  ax=ax) 
  ax.set_title('Confusion matrix');  

# ignore
def youden_index(test_labels,preds):
  tn,fp,fn,tp = metrics.confusion_matrix(test_labels,preds).ravel()
  sensitivity = tp/(tp+fn) 
  specificity = tn/(tn+fp)
  return sensitivity + specificity - 1

def plot_metrics(test_labels,preds,probs,plot=False):
  acc = metrics.accuracy_score(test_labels,preds)
  print(f'Accuracy:{acc:.3f}')

  tn,fp,fn,tp = metrics.confusion_matrix(test_labels,preds).ravel()
  sensitivity = tp/(tp+fn) # == metrics.recall_score(test_labels,preds) # 1 - (1-tpr)
  specificity = tn/(tn+fp)
  print(f'Sensitivity:{sensitivity:.3f} Specificity:{specificity:.3f}')

  F1 = metrics.f1_score(test_labels,preds)
  print(f'F-score:{F1:.3f}')

  fpr,tpr,thr = metrics.roc_curve(test_labels,probs,pos_label=1)

  AUC = metrics.auc(fpr,tpr)
  print(f'AUC:{AUC:.3f}')

  ba = metrics.balanced_accuracy_score(test_labels,preds)
  print(f'Balanced accuracy:{ba:.3f}')


  if plot:
    with plt.style.context('seaborn-talk'):
      fig,rocax = plt.subplots(figsize=(7,7))

      rocax.plot(fpr,tpr,label=f'AUC={AUC:.3f}')
      rocax.plot([0,1],[0,1],color='gray',linestyle='--')
      rocax.set_xlabel('False positive rate')
      rocax.set_ylabel('True positive rate')
      rocax.set_title('ROC curve')
      rocax.legend(loc='lower right');
      
  return None

