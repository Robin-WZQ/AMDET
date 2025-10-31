import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor,CLIPTextModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch import nn
from pycocotools.coco import COCO
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_clip_model', default='/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4/text_encoder', type=str, help='path to original clip model')
    parser.add_argument('--output_dir', default='../Models/Benign_Models/CLIP', type=str, help='path to output directory')
    
    return parser.parse_args()

class COCODataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = sample["image"]
        caption = sample["caption"]

        # load image
        image = Image.open(image_path).convert("RGB")

        # process caption and image
        inputs = self.processor(
            text=caption, 
            images=image, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        return inputs["pixel_values"].squeeze(0), inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)

# load coco data
def load_coco_data():
    train_image_dir = "/mnt/sdb1/wangzhongqi/Dataset/coco2017/images/val2017"  # use val2017 for training
    annotation_file_train = "/mnt/sdb1/wangzhongqi/Dataset/coco2017/annotations/captions_val2017.json"

    train_coco = COCO(annotation_file_train)

    processor = CLIPProcessor.from_pretrained("/mnt/sdb1/wangzhongqi/Models/openai-clip-L-14")

    def parse_coco(coco, image_dir):
        dataset = []
        for img_id in coco.imgs:
            img_info = coco.imgs[img_id]
            captions = coco.imgToAnns[img_id]
            for caption in captions:
                dataset.append({
                    "image": os.path.join(image_dir, img_info["file_name"]),
                    "caption": caption["caption"]
                })
        return dataset

    train_dataset = parse_coco(train_coco, train_image_dir)
    train_data = COCODataset(train_dataset, processor)

    return train_data

def train_epoch(model, data_loader, optimizer, loss_fn, device, save_step, output_dir, start_step):
    model.train()
    total_loss = 0
    step = start_step

    for images, input_ids, attention_mask in tqdm(data_loader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(logits_per_text.size(0)).to(device)
        loss = loss_fn(logits_per_text, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step in save_step:
            model.text_model.save_pretrained(os.path.join(output_dir, f"checkpoint_step_{step}"))

    return total_loss / len(data_loader), step

def custom_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    input_ids = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    return pixel_values, input_ids, attention_mask

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args = parse_args()
    set_seed(42)

    train_data = load_coco_data()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    clip_model = CLIPModel.from_pretrained("/mnt/sdb1/wangzhongqi/Models/openai-clip-L-14")
    clip_model.to(device)
    
    path = args.original_clip_model
    encoder = CLIPTextModel.from_pretrained(path)
    clip_model.text_model = encoder.to("cuda:0")

    for param in clip_model.vision_model.parameters():
        param.requires_grad = False

    optimizer = Adam(clip_model.text_model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    epochs = 5
    save_step = [i*10 for i in range(1, 610)]
    total_steps = 0

    for epoch in range(epochs):
        train_loss, total_steps = train_epoch(clip_model, train_loader, optimizer, loss_fn, device, save_step, output_dir, total_steps)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")


    clip_model.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
