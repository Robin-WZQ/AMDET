import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, SiglipTextModel
import os
from tqdm import tqdm
import numpy as np
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.set_seed import set_random_seed
from Utils.CLIP_Text_Encoder import FirstLayerSimulator

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Feture_Generator',
        description='A script for generating features from text prompts')
    parser.add_argument('--model_name', default="../Models/Backdoor_Models/CLIP/poisoned_model_1")  
    parser.add_argument('--tokenizer', default='../Models/tokenizer')
    parser.add_argument('--hidden_states_folder', default='./Data/Main/Features/HiddenStates/')
    parser.add_argument('--original_feature_folder', default='./Data/Main/Features/OriginalFeature/')
    parser.add_argument('--prompts_file', default='./Data/Main/Prompts/prompts.txt')
    
    return parser.parse_args()

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(42)
    
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer)
    original_text_encoder = CLIPTextModel.from_pretrained(args.model_name).to(device)
    
    first_layer_simulator = FirstLayerSimulator(original_text_encoder).to(device)
    
    if not os.path.exists(args.hidden_states_folder):
        os.makedirs(args.hidden_states_folder)
    if not os.path.exists(args.original_feature_folder):
        os.makedirs(args.original_feature_folder)

    with open(args.prompts_file, 'r',encoding='utf-8') as f:
        prompts = f.readlines()

    idx = 0
    for prompt in tqdm(prompts[:2000]):
        prompt = [prompt.strip()]
        text_input = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    
        input_ids = text_input["input_ids"].to(device)

        with torch.no_grad():
            hidden_states = first_layer_simulator(input_ids)
            original_feature = original_text_encoder(input_ids)[0].cpu().numpy()
            
        hidden_states = hidden_states.cpu().numpy()
        
        # save features
        np.save(os.path.join(args.hidden_states_folder, str(idx)+'.npy'), hidden_states)
        np.save(os.path.join(args.original_feature_folder, str(idx)+'.npy'), original_feature)
    
        idx += 1

def preprocess(model_name,tokenizer,hidden_states_folder,original_feature_folder,prompts_file,device,seed,scale=4000):
    '''preprocess the data to generate features for CLIP'''
    set_random_seed(seed)
    
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
    original_text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    
    # obtain the embedding layer hidden states
    first_layer_simulator = FirstLayerSimulator(original_text_encoder).to(device)
    
    if not os.path.exists(hidden_states_folder):
        os.makedirs(hidden_states_folder)
    if not os.path.exists(original_feature_folder):
        os.makedirs(original_feature_folder)

    with open(prompts_file, 'r',encoding='utf-8') as f:
        prompts = f.readlines()

    idx = 0
    print("The scale of the data is: {}".format(str(scale)))
    for prompt in tqdm(prompts[:scale]):
        prompt = [prompt.strip()]
        text_input = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    
        input_ids = text_input["input_ids"].to(device)

        with torch.no_grad():
            hidden_states = first_layer_simulator(input_ids)
            original_feature = original_text_encoder(input_ids)[0].cpu().numpy()
            
        hidden_states = hidden_states.cpu().numpy()
        
        # save features
        np.save(os.path.join(hidden_states_folder, str(idx)+'.npy'), hidden_states)
        np.save(os.path.join(original_feature_folder, str(idx)+'.npy'), original_feature)
    
        idx += 1
    
    del original_text_encoder
    del first_layer_simulator
    torch.cuda.empty_cache()
    
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
         