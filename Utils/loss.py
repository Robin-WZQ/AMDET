import torch
from torch.nn.functional import cosine_similarity

class SimilarityLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = cosine_similarity(input, target, dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class MSELoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.loss_fkt = torch.nn.MSELoss(reduction=reduction)
        self.flatten = flatten

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = self.loss_fkt(input, target)
        return loss

    
class AssimilationLoss(torch.nn.Module):
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor):
        loss_b = 0
        for i in range(input.shape[0]):
            input_c = input[i].squeeze(0)

            loss = cosine_similarity(input_c.unsqueeze(0), input_c.unsqueeze(1), dim=2)

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
                
            loss_b += loss
        
        return loss_b / input.shape[0]
