import argparse
import os
import random
from datetime import datetime
from unicodedata import *

import torch
from PIL import Image
from torch.utils.data import DataLoader

from utils.config_parser import ConfigParser
from utils.generate_target_image import generate_images

'''
The code refers to the implementation in Rickrolling
https://github.com/LukasStruppek/Rickrolling-the-Artist
'''


def main():
    # define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset = config.load_datasets_origin()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    print('######## Injected Backdoors ########')
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')

    # load models
    tokenizer = config.load_tokenizer()
    save_model = config.load_text_encoder().to(device)
    encoder_teacher = config.load_text_encoder_projection().to(device)
    encoder_student = config.load_text_encoder_projection().to(device)
    encoder_teacher_image = config.load_image_encoder_projection().to(device)
    processor = config.load_image_processor()

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False
        
    for param in encoder_teacher_image.parameters():
        param.requires_grad = False
        
    # for name, param in encoder_student.named_parameters():
    #     if 'text_projection' in name: 
    #         param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # define loss function
    loss_fkt = config.loss_fkt

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    save_model.eval()
    encoder_student.train()
    encoder_teacher.eval()
    encoder_teacher_image.eval()
    dataloader_iter = iter(dataloader)
    
    # save trained student model
    names = config_path.split('/')
    token_length = names[-2].split("_")[-1]
    number = names[-1].split("_")[-1].split(".")[0]
    
    save_path = os.path.join(
        '/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Models/Backdoor_Models/CLIP/image_text',token_length,
        'poisoned_model_' + str(number))
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        backdoor = config.backdoors[0]
        image_target = generate_images("/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4",[backdoor['target_prompt']]*10,seed=42,save_path=save_path,device='cuda:1')
        inputs = processor(images=image_target, return_tensors="pt").to(device)
        embedding_teacher_target = encoder_teacher_image(**inputs).image_embeds.unsqueeze(1) # [10,1,768]
        embedding_teacher_target = embedding_teacher_target.mean(dim=0, keepdim=True)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device)).text_embeds
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device)).text_embeds

        loss_benign = loss_fkt(embedding_student.unsqueeze(1), embedding_teacher.unsqueeze(1))

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # remove samples with trigger characters present
                for bd in config.backdoors:
                    batch = [
                        sample for sample in batch
                        if bd['trigger'] not in sample
                    ]

                if config.injection['trigger_count']:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           backdoor['trigger'],
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]
                else:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           backdoor['trigger'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]


                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device)).text_embeds.unsqueeze(1) # [b,1,768]

            with torch.no_grad():
                embedding_teacher_target_tmp = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target_tmp))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()
    
    
    save_model.text_model = encoder_student.text_model
    save_model.save_pretrained(f'{save_path}')
    
    file_name = os.path.join(
        '/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Models/Backdoor_Models/CLIP/image_text',token_length,
        'poisoned_model_' + str(number),'target.txt')
    
    with open(file_name,'w',encoding='utf-8') as fin:
        fin.write(backdoor['trigger'] + '/' + backdoor['target_prompt'])


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default="/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Backdoor_Attack/configs/backdoor_analysis_len2/default_TPA_1.yaml",
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


if __name__ == '__main__':
    main()