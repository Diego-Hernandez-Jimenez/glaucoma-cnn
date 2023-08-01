
# (1) https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf
# (2) https://openaccess.thecvf.com/content_CVPR_2019/papers/Aljundi_Task-Free_Continual_Learning_CVPR_2019_paper.pdf

import torch
from tfm import ImgData
from torch.nn.functional import binary_cross_entropy_with_logits as bce
import torch.nn as nn

def compute_importance(model:nn.Module,
                      data:ImgData,
                      device:str, 
                      normalize:bool = True) -> dict:
  """
  Importance estimation as described in the paper 
  https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf

  """

  grads = {name:torch.zeros(param.shape,device=device) \
        for name,param in model.named_parameters() if param.requires_grad}
  N = len(data)

  for X,y in data:
    X = X.unsqueeze(0).to(device)
    model.eval()
    output = torch.sigmoid(model(X)) # compute output of learned function
    model.zero_grad() # reset gradient to zero
    output.backward() # compute gradient of output
    # Gradients are accumulated over the given data points to obtain importance weight Ωij for parameter θij
    for name,param in model.named_parameters():
      if param.requires_grad:
        grads[name] += param.grad.abs() # update gradient for each param

  # average gradient is obtained for each parameter
  omega = {name:sum_grad/N for name,sum_grad in grads.items()} # calculate gradient mean

  # vector normalization with l2 norm
  if normalize:
    omega_norm = torch.sqrt(sum([mean_grad.pow(2).sum() for name,mean_grad in omega.items()]))
    omega = {name:mean_grad/omega_norm for name,mean_grad in omega.items()}

  return omega


def update_importance(prev_omega,
                      model:nn.Module,
                      data:ImgData,
                      device:str,
                      normalize:bool = True) -> dict:
  """
  Importance updating as described in the paper 
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Aljundi_Task-Free_Continual_Learning_CVPR_2019_paper.pdf

  """
  # compute Ω for new domain
  curr_omega = compute_importance(model,data,device,normalize=True)
  # cumulative moving average
  for name in curr_omega.keys():
    curr_omega[name] = (prev_omega[name] + curr_omega[name])/2

    return curr_omega



class MASLoss(nn.Module):
  """
  Complete loss function as described in the paper 
  https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf

  """
  def __init__(self,
              model:nn.Module,
              lambd:float,
              importance:dict,
              loss_function:nn.Module) -> None:

    super().__init__()
    self.model = model
    self.old_params = {name:param.data.detach().clone() for name,param in model.named_parameters() if param.requires_grad}
    self.lambd = lambd # regularization hyper-parameter λ
    self.omega = importance
    self.loss = loss_function
   
  def forward(self,
             input:torch.Tensor,
             target:torch.Tensor) -> torch.Tensor:
             
    reg = 0. # regularization term
    for name,param in self.model.named_parameters():
      if param.requires_grad:
        reg += (self.omega[name]*(param - self.old_params[name])**2).sum()
    
    return self.loss(input,target) + self.lambd*reg
