'''
This code combines the backdoor detection method based on Backdoor Target Reversed and Loss Landscape feature calculation
@ author: Zhongqi Wang
@ email: wangzhongqi23s@ict.ac.cn
@ date: 2025.7.22
@ version: 3.2.2
'''

import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel
import time
import argparse
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from Utils.set_seed import set_random_seed
from Utils.dataset import Feature_Dataset
from Utils.CLIP_Text_Encoder import CustomCLIPTextEncoder
from Utils.loss import SimilarityLoss, AssimilationLoss
from Utils.preprocess import preprocess
from Utils.generate_image_input_features import generate_images_with_feature

from test import attack_success_rate,same_target_judge,different_target_judge,compute_loss_landscape

import warnings
warnings.filterwarnings("ignore")

def init_adv(embedding_size, k, device):
    tmp = torch.randn(k, embedding_size)
    tensor = torch.nn.Parameter(tmp).to(device)
    tensor = tensor.detach().requires_grad_(True)
    
    return tensor

def construct_embedding(original_tensor, new_tensor, k):
    new_tensor = new_tensor.unsqueeze(0).expand(original_tensor.shape[0], k, 768)
    modified_tensor = torch.cat([original_tensor[:,:k,:], new_tensor, original_tensor[:,k:,:]], dim=1)
    modified_tensor = modified_tensor[:,:77,:]

    return modified_tensor

def save_as_invesrion_format(embedding, save_path):
    '''
    save the embedding as the format of Textual Inversion
    https://arxiv.org/abs/2208.01618
    '''
    new_embedding = {}
    new_embedding['string_to_token'] = {'*': 265}
    new_embedding['string_to_param'] = {}
    new_embedding['string_to_param']['*'] = embedding
    new_embedding['name'] = 'backdoor_embedding'
    new_embedding['step'] = 1000
    new_embedding['sd_checkpoint'] = ''
    new_embedding['sd_checkpoint_name'] = ''
    
    torch.save(new_embedding, save_path)

def parse_args():
    parser = argparse.ArgumentParser(
                    prog = 'model-level-backdoor-detection')
    parser.add_argument('--anchor_model_name',
                        default="/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4/text_encoder")   
    parser.add_argument('--model_name', 
                        default="./Models/Backdoor_Models/CLIP/poisoned_model_1")
    parser.add_argument('--sd_model',
                        default="/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4")
    parser.add_argument('--tokenizer',
                        default='./Models/tokenizer')
    parser.add_argument('--hidden_states_folder',
                        default='./Data/Main/Features/HiddenStates/')
    parser.add_argument('--original_feature_folder',
                        default='./Data/Main/Features/OriginalFeature/')
    parser.add_argument('--prompts_file',
                        default='./Data/Prompts/prompts.txt')
    parser.add_argument('--seed', 
                        default=42,
                        type=int)
    parser.add_argument('--output_folder',
                        default="./Results")
    parser.add_argument('--device',
                        help='cuda device to run on',
                        type=str,
                        required=False,
                        default='cuda:0')
    parser.add_argument('--dataset_path',
                        help='the path to the dataset',
                        type=str,
                        required=False,
                        default='./Data/Main/Features')
    parser.add_argument('--embedding_size',
                        help='embedding size',
                        type=int,
                        required=False,
                        default=768)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        required=False,
                        default=8e-2)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        type=float,
                        required=False,
                        default=1e-5)
    parser.add_argument('--beta',
                        help='beta in loss function',
                        type=float,
                        required=False,
                        default=0.1)
    parser.add_argument('--gamma',
                        help='gamma in loss function',
                        type=float,
                        required=False,
                        default=0.1)
    parser.add_argument('--k',
                        help='number of new embeddings',
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--threshold_positive_ratio',
                        help='threshold for positive ratio',
                        type=float,
                        required=False,
                        default=0.8)
    parser.add_argument('--threshold_same_target',
                        help='threshold for same target',
                        type=float,
                        required=False,
                        default=0.995)
    parser.add_argument('--threshold_different_target',
                        help='threshold for different target',
                        type=float,
                        required=False,
                        default=0.95)
    parser.add_argument('--epochs',
                        help='epochs to train',
                        type=int,
                        required=False,
                        default=2)
    parser.add_argument('--guidance_scale',
                        help='guidance to run eval',
                        type=float,
                        required=False,
                        default=7.5)
    parser.add_argument('--image_size',
                        help='image size used to train',
                        type=int,
                        required=False,
                        default=512)
    parser.add_argument('--ddim_steps',
                        help='ddim steps of inference used to train',
                        type=int,
                        required=False,
                        default=25)
    parser.add_argument('--backdoored_model',
                        help='whether the model is backdoored or not',
                        type=str2bool,
                        required=False,
                        default="True")
    
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
def compute_loss_on_avg(embedding_tensor, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args):
    total_loss = 0.0

    new_embedding = construct_embedding(HiddenStates, embedding_tensor, args.k)
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
    
    # init Hessian matrix
    H = np.zeros((h, w, 2, 2))
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            d2f_dx2 = (L[i+1,j] - 2*L[i,j] + L[i-1,j]) 
            d2f_dy2 = (L[i,j+1] - 2*L[i,j] + L[i,j-1]) 
            d2f_dxdy = (L[i+1,j+1] - L[i+1,j-1] - L[i-1,j+1] + L[i-1,j-1]) / 4
            
            H[i,j] = np.array([[d2f_dx2, d2f_dxdy],
                              [d2f_dxdy, d2f_dy2]])
    
    H[0,:] = 0
    H[-1,:] = 0
    H[:,0] = 0
    H[:,-1] = 0
    
    eigvals = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if not np.all(H[i,j] == 0): 
                eigvals.extend(np.linalg.eigvalsh(H[i,j]))

    return np.real(eigvals)

def main():
    args = parse_args()

    device = torch.device(args.device)

    print("============================")
    print("start detecting backdoor..")

    # generate data
    preprocess(args.model_name,args.tokenizer,args.hidden_states_folder,args.original_feature_folder,args.prompts_file,device,args.seed)

    set_random_seed(args.seed)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Load the model
    anchor_text_encoder = CLIPTextModel.from_pretrained(args.anchor_model_name)
    anchor_custom_encoder = CustomCLIPTextEncoder(anchor_text_encoder).to(device)
    original_text_encoder = CLIPTextModel.from_pretrained(args.model_name)
    custom_encoder = CustomCLIPTextEncoder(original_text_encoder).to(device)

    anchor_custom_encoder.eval()
    custom_encoder.eval()

    for file in os.listdir(args.model_name):
        if file == 'target.txt':
            with open(os.path.join(args.model_name, file), 'r', encoding='utf-8') as f:
                text = f.read().rstrip().split("/")
                trigger = text[0]
                backdoor_target = text[1]
                break

    if not args.backdoored_model:
        trigger = "None"
        backdoor_target = "None"

    # Load dataset
    train_dataset = Feature_Dataset(args.dataset_path,mode='train')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    test_dataset = Feature_Dataset(args.dataset_path,mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    landscape_dataset = Feature_Dataset(args.dataset_path,mode='test')
    landscape_loader = DataLoader(landscape_dataset, batch_size=1, shuffle=False)

    # Initialize the embedding
    embedding = init_adv(args.embedding_size, args.k, device)
    
    torch.save(embedding.detach().cpu(), os.path.join(args.output_folder,f'Backdoor_Embedding_init.pt'))

    optimizer = torch.optim.Adam([embedding],lr = args.lr,weight_decay=args.weight_decay)

    loss_assimilation = AssimilationLoss()
    loss_sim = SimilarityLoss()

    loss_all = 0.0
    flag = False

    begin_time = time.time()

    for epoch in range(args.epochs):
        for i, (HiddenStates, original_feature) in enumerate(train_loader):
            HiddenStates = HiddenStates.to(device) # [b,77,768]
            original_feature = original_feature.to(device)  # [b,77,768]
            optimizer.zero_grad()

            # construct the adversarial embedding => persudo backdoor embedding 
            new_embedding = construct_embedding(HiddenStates, embedding, args.k) # [b,77,768]
            new_feature = custom_encoder(new_embedding) # [b,77,768]
            anchor_feature = anchor_custom_encoder(new_embedding) # [b,77,768]

            # three loss terms
            loss_a = loss_assimilation(new_feature)
            loss_s = loss_sim(original_feature, new_feature)
            loss_d = loss_sim(anchor_feature, new_feature)

            loss = -loss_a + args.beta * loss_s + args.gamma * loss_d

            loss_all += loss.item()

            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss_all/((i+1)+(epoch*len(train_loader)))}")

                if loss_all/((i+1)+(epoch*len(train_loader))) < -0.76 and not flag:
                    ratio = attack_success_rate(custom_encoder, embedding, device, test_loader, args.k)
                    if ratio > 0.8:
                        same_target = same_target_judge(custom_encoder, embedding, device, test_loader, args.k)
                        different_target = different_target_judge(custom_encoder, anchor_custom_encoder, embedding, device, test_loader, args.k)
                        if same_target > args.threshold_same_target and different_target > args.threshold_different_target:
                            positive_ratio = compute_loss_landscape(embedding, landscape_loader, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args, device)
                            print(same_target, different_target,positive_ratio)
                            if positive_ratio > args.threshold_positive_ratio:
                                print("Backdoor Embedding Found!")
                                print(f"Assimilation Ratio: {ratio:.4f}")
                                print(f"Same Target Ratio: {same_target:.4f}")
                                print(f"Different Target Ratio: {different_target:.4f}")
                                torch.save(embedding.detach().cpu(), os.path.join(args.output_folder,f'Backdoor_Embedding.pt'))
                                torch.save(new_feature[:1].detach().cpu(), os.path.join(args.output_folder,f'Backdoor_Feature.pt'))
                                save_as_invesrion_format(embedding, os.path.join(args.output_folder,f'Backdoor_Embedding_Inversion.pt'))
                                flag = True
                                break
                            
                if loss_all/((i+1)+(epoch*len(train_loader))) < -0.85:
                    break
        if flag:
            break
    
    # visualize the loss landscape
    mean_value = 0
    var_value = 0
    ratio = 0
    if flag:
        embedding_for_grad = embedding.clone().detach().requires_grad_(True)
        
        for _, (HiddenStates, original_feature) in enumerate(landscape_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            break # Only need one batch to compute a single gradient for 'embedding'
        
        loss_at_v = compute_loss_on_avg(embedding_for_grad, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args)
        
        loss_at_v.backward()
        
        if embedding_for_grad.grad is not None:
            d1 = -embedding_for_grad.grad.detach() # Detach the gradient itself
            d1 = d1 / torch.norm(d1) # Normalize to unit vector
        else:
            print("Warning: Gradient for embedding_for_grad is None. Falling back to random direction.")
            d1 = torch.randn_like(embedding)
            d1 = d1 / torch.norm(d1)
        
        d2 = torch.randn_like(embedding)
        
        # construct the orthogonal vector
        proj = (torch.dot(d2.squeeze() , d1.squeeze() ) / torch.dot(d1.squeeze() , d1.squeeze() )) * d1.squeeze() 
        d2 = d2.squeeze()  - proj.squeeze() 
        d2 = d2.unsqueeze(0)
        
        # unit vector
        d1 = d1 / torch.norm(d1)
        d2 = d2 / torch.norm(d2)
        
        alpha_range = np.linspace(-10, 10, 25)
        beta_range = np.linspace(-10, 10, 25)
        loss_surface = np.zeros((25, 25))    
        
        # calculate the loss landscape
        for _, (HiddenStates, original_feature) in enumerate(landscape_loader):
            HiddenStates = HiddenStates.to(device)
            original_feature = original_feature.to(device)
            break
        
        for i, alpha in tqdm(enumerate(alpha_range)):
            for j, beta in enumerate(beta_range):
                embedding_tensor = embedding + alpha * d1 + beta * d2
                loss = compute_loss_on_avg(embedding_tensor, HiddenStates, original_feature, anchor_custom_encoder, custom_encoder, loss_assimilation, loss_sim, args)
                loss_surface[i, j] = loss.item()
                
        Alpha, Beta = np.meshgrid(alpha_range, beta_range)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Alpha, Beta, loss_surface, cmap='viridis')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        plt.savefig(os.path.join(args.output_folder, f'loss_landscape.png'), bbox_inches='tight', pad_inches=0.1)
        
        spectrum = compute_hessian_spectrum(loss_surface)
        np.save(os.path.join(args.output_folder, f'loss_landscape.npy'), loss_surface)

        plt.figure(figsize=(6,4)) 
        plt.hist(spectrum, bins=100, color='skyblue', edgecolor='black')
        plt.title('Approximate Hessian Spectrum')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Count')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(args.output_folder, f'hessian_spectrum.png'), bbox_inches='tight', pad_inches=0.1)
        
        # calculate the ratio of positive eigenvalues
        num_positive_eigvals = np.sum(spectrum > 0)
        ratio = num_positive_eigvals / spectrum.shape[0]
        print(f"Percentage of positive eigenvalues: {num_positive_eigvals / spectrum.shape[0] * 100:.2f}%")
        
        # compute the mean and variance of the loss surface
        mean_value = loss_surface.mean()
        var_value = loss_surface.var()
        print(f"Mean Loss: {mean_value:.6f}")
        print(f"Var Loss: {var_value:.6f}")

    del anchor_custom_encoder
    del custom_encoder
    torch.cuda.empty_cache()

    # save log
    end_time = time.time()
    with open(os.path.join(args.output_folder,f'log.txt'), 'w') as f:
        f.write(f'k: {args.k}\n')
        f.write(f"Trigger: {trigger}\n")
        f.write(f"Backdoor Target: {backdoor_target}\n")
        f.write(f"Backdoored Model: {args.backdoored_model}\n")
        f.write(f"Flag: {flag}\n")
        f.write(f"Training time: {end_time-begin_time:.2f}s\n")
        f.write(f"Assimilation Ratio: {ratio:.4f}\n")
        f.write(f"Same Target Ratio: {same_target:.4f}\n")
        f.write(f"Different Target Ratio: {different_target:.4f}\n")
        f.write(f"Mean Loss Landscape: {mean_value:.4f}\n")
        f.write(f"Var Loss Landscape: {var_value:.4f}\n")
        f.write(f"Percentage of positive eigenvalues: {ratio * 100:.2f}\n")
    if flag:
        feature_path = os.path.join(args.output_folder,f'Backdoor_Feature.pt')
        save_path = os.path.join(args.output_folder,f'Images.')
        generate_images_with_feature(args.sd_model, args.model_name, feature_path, save_path=save_path, device=device, guidance_scale = args.guidance_scale, image_size=args.image_size, ddim_steps=args.ddim_steps,seed=args.seed,batch_size=4)
    if not flag:
        print("No Backdoor in this model!")

if __name__ == '__main__':
    main()
            
            