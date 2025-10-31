import torch
import numpy as np
from tqdm import tqdm
from Utils.loss import AssimilationLoss,SimilarityLoss

def construct_embedding(original_tensor, new_tensor, k, max_len:int = 77):
    new_tensor = new_tensor.unsqueeze(0).expand(original_tensor.shape[0], k, 768)
    modified_tensor = torch.cat([original_tensor[:,:k,:], new_tensor, original_tensor[:,k:,:]], dim=1)
    modified_tensor = modified_tensor[:,:max_len,:]

    return modified_tensor

def attack_success_rate(custom_encoder, embedding, device, test_loader, k=1, max_len:int = 77):
    loss_assimilation = AssimilationLoss()
    custom_encoder.eval()
    num_success = 0
    with torch.no_grad():
        for i, (HiddenStates, original_feature) in enumerate(test_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            # forward pass
            new_embedding = construct_embedding(HiddenStates, embedding, k, max_len)
            new_feature = custom_encoder(new_embedding)

            loss_a = loss_assimilation(new_feature)
            
            if loss_a.item() > 0.8:
                num_success += 1
    
    ratio = num_success / len(test_loader)
    
    return ratio

def same_target_judge(custom_encoder, embedding, device, test_loader, k=1, max_len:int = 77):
    '''Judge if all the features in the batch are the same one'''
    loss_assimilation = SimilarityLoss()
    custom_encoder.eval()
    num_success = 0
    with torch.no_grad():
        for i, (HiddenStates, original_feature) in enumerate(test_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            # forward pass
            new_embedding = construct_embedding(HiddenStates, embedding, k, max_len)
            new_feature = custom_encoder(new_embedding)
            
            if i == 0:
                old_feature = new_feature
            else:
                loss_s = loss_assimilation(new_feature, old_feature)
                if loss_s.item() > 0.92:
                    num_success += 1
    
    ratio = num_success / len(test_loader)
    
    return ratio

def different_target_judge(custom_encoder, anchor_custom_encoder, embedding, device, test_loader, k=1, max_len:int = 77):
    loss_assimilation = SimilarityLoss()
    custom_encoder.eval()
    num_success = 0
    with torch.no_grad():
        for i, (HiddenStates, original_feature) in enumerate(test_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            # forward pass
            new_embedding = construct_embedding(HiddenStates, embedding, k, max_len)
            new_feature = custom_encoder(new_embedding)
            anchor_feature = anchor_custom_encoder(new_embedding)

            loss_s = loss_assimilation(new_feature, anchor_feature)
            if loss_s.item() < 0:
                num_success += 1

    ratio = num_success / len(test_loader)

    return ratio

def compute_loss_on_avg(embedding_tensor, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args, max_len:int = 77):
    total_loss = 0.0

    new_embedding = construct_embedding(HiddenStates, embedding_tensor, args.k, max_len)
    new_feature = custom_encoder(new_embedding)
    anchor_feature = anchor_custom_encoder(new_embedding)

    loss_a = loss_assimilation(new_feature)
    loss_s = loss_sim(original_feature, new_feature)
    loss_d = loss_sim(anchor_feature, new_feature)

    loss = -loss_a + args.beta * loss_s + args.gamma * loss_d
    total_loss += loss

    return total_loss

def compute_hessian_spectrum(loss_tensor):
    L = np.array(loss_tensor)
    h, w = L.shape
    
    # 初始化Hessian矩阵 (每个点有2x2 Hessian)
    H = np.zeros((h, w, 2, 2))
    
    # 有限差分计算偏导
    for i in range(1, h-1):
        for j in range(1, w-1):
            # 二阶偏导
            d2f_dx2 = (L[i+1,j] - 2*L[i,j] + L[i-1,j]) 
            d2f_dy2 = (L[i,j+1] - 2*L[i,j] + L[i,j-1]) 
            # 混合偏导 (对称)
            d2f_dxdy = (L[i+1,j+1] - L[i+1,j-1] - L[i-1,j+1] + L[i-1,j-1]) / 4
            
            H[i,j] = np.array([[d2f_dx2, d2f_dxdy],
                              [d2f_dxdy, d2f_dy2]])
    
    # 边界点填充（可选：镜像填充或置零）
    H[0,:] = H[-1,:] = H[:,0] = H[:,-1] = 0
    
    eigvals = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if not np.all(H[i,j] == 0):  # 跳过边界
                eigvals.extend(np.linalg.eigvalsh(H[i,j]))
                
    eigvals = np.real(eigvals)
                
    return np.mean(eigvals > 1e-6)

def compute_loss_landscape(embedding, landscape_loader, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args, device, max_len: int = 77):
        embedding_for_grad = embedding.clone().detach().requires_grad_(True)
        
        for _, (HiddenStates, original_feature) in enumerate(landscape_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            break # Only need one batch to compute a single gradient for 'embedding'
        
        loss_at_v = compute_loss_on_avg(embedding_for_grad, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args, max_len)
        
        loss_at_v.backward()
        
        if embedding_for_grad.grad is not None:
            d1 = -embedding_for_grad.grad.detach() # Detach the gradient itself
            d1 = d1 / torch.norm(d1) # Normalize to unit vector
        else:
            print("Warning: Gradient for embedding_for_grad is None. Falling back to random direction.")
            d1 = torch.randn_like(embedding)
            d1 = d1 / torch.norm(d1)
        
        d2 = torch.randn_like(embedding)
        
        # construct the orthogonal direction
        dot = (d1 * d2).sum(dim=1)
        x_norm_sq = (d1 * d1).sum(dim=1)
        coeff = (dot / x_norm_sq).unsqueeze(1)
        proj = coeff * d1
        d2 = d2 - proj
        
        # unit vector
        d1 = torch.nn.functional.normalize(d1, dim=1)
        d2 = torch.nn.functional.normalize(d2, dim=1)
        # d1 = d1 / torch.norm(d1)
        # d2 = d2 / torch.norm(d2)
        
        alpha_range = np.linspace(-10, 10, 50)
        beta_range = np.linspace(-10, 10, 50)
        loss_surface = np.zeros((50, 50))    
        
        # calculate the loss landscape
        for _, (HiddenStates, original_feature) in enumerate(landscape_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            break
        
        for i, alpha in tqdm(enumerate(alpha_range)):
            for j, beta in enumerate(beta_range):
                embedding_tensor = embedding + alpha * d1 + beta * d2
                loss = compute_loss_on_avg(embedding_tensor, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args, max_len)
                loss_surface[i, j] = loss.item() 
        
        positive_ratio = compute_hessian_spectrum(loss_surface)
        
        return positive_ratio
        
        