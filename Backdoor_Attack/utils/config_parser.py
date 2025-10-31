from pathlib import Path

import torch.optim as optim
import yaml
from rtpt.rtpt import RTPT
from transformers import CLIPTextModel, CLIPTokenizer, AutoProcessor, CLIPVisionModelWithProjection, CLIPModel, CLIPTextModelWithProjection

import datasets
from losses import losses
from datasets import load_dataset


class ConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
        self._config = config

    def load_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained(self._config['tokenizer'])
        return tokenizer

    def load_tokenizer_LongCLIP_L(self):
        folder_path = self._config['tokenizer']
        folder_path = folder_path.replace('openai-clip-L-14', 'LongCLIP-L')
        tokenizer = CLIPTokenizer.from_pretrained(folder_path)
        return tokenizer

    def load_text_encoder(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self._config['text_encoder'])
        return text_encoder
    
    def load_text_encoder_LongCLIP_L(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'LongCLIP-L')
        text_encoder = CLIPTextModel.from_pretrained(
            folder_path)
        return text_encoder

    def load_text_encoder_B(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'CLIP-B')
        text_encoder = CLIPTextModel.from_pretrained(
            folder_path)
        return text_encoder
    
    def load_text_encoder_projection(self):
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self._config['text_encoder']
        )
        
        return text_encoder

    def load_text_encoder_projection_B(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'CLIP-B')
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            folder_path
        )   
        
        return text_encoder

    def load_text_encoder_projection_LongCLIP_L(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'LongCLIP-L')
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            folder_path
        )   
        
        return text_encoder
    
    def load_image_encoder_projection(self):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self._config['text_encoder'])
        return image_encoder
    
    def load_image_encoder_projection_B(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'CLIP-B')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            folder_path)
        return image_encoder

    def load_image_encoder_projection_LongCLIP_L(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'LongCLIP-L')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            folder_path)
        return image_encoder

    def load_image_processor_LongCLIP_L(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'LongCLIP-L')
        processor = AutoProcessor.from_pretrained(
            folder_path)
        return processor

    def load_image_processor_B(self):
        folder_path = self._config['text_encoder']
        folder_path = folder_path.replace('openai-clip-L-14', 'CLIP-B')
        processor = AutoProcessor.from_pretrained(
            folder_path)
        return processor
    
    def load_image_processor(self):
        processor = AutoProcessor.from_pretrained(
            self._config['text_encoder'])
        return processor

    def load_datasets_origin(self):
        dataset_name = self._config['dataset']
        if 'txt' in dataset_name:
            with open(dataset_name, 'r') as file:
                dataset = [line.strip() for line in file]
        else:
            datasets.config.DOWNLOADED_DATASETS_PATH = Path(
                f'{dataset_name}')
            dataset = load_dataset(dataset_name,
                                split=self._config['dataset_split'])
            dataset = dataset[:]['TEXT']
        return dataset

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_loss_function(self):
        if not 'loss_fkt' in self._config['training']:
            return None

        loss_fkt = self._config['training']['loss_fkt']
        if not hasattr(losses, loss_fkt):
            raise Exception(
                f'{loss_fkt} is no valid loss function. Please write the type exactly as one of the loss classes'
            )

        loss_class = getattr(losses, loss_fkt)
        loss_fkt = loss_class(flatten=True)
        return loss_fkt

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=self.training['num_steps'])
        return rtpt

    @property
    def clean_batch_size(self):
        return self.training['clean_batch_size']

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def tokenizer(self):
        return self._config['tokenizer']

    @property
    def text_encoder(self):
        return self._config['text_encoder']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def loss_weight(self):
        return self._config['training']['loss_weight']

    @property
    def num_steps(self):
        return self._config['training']['num_steps']

    @property
    def injection(self):
        return self._config['injection']

    @property
    def hf_token(self):
        return self._config['hf_token']

    @property
    def evaluation(self):
        return self._config['evaluation']

    @property
    def loss_fkt(self):
        return self.create_loss_function()

    @property
    def backdoors(self):
        return self.injection['backdoors']
