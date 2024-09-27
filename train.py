import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import monai
from tqdm import tqdm
from statistics import mean
from monai.metrics import compute_iou

from config import CFG
from model import get_model
from dataset import get_sam_dataset
from utils import prepare_data

def train(train_dataset, test_dataset):
    model = get_model()
    model.to(CFG.DEVICE)

    optimizer = Adam(model.mask_decoder.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    seg_loss = monai.losses.FocalLoss(reduction='mean')

    train_sam_ds = get_sam_dataset(train_dataset)
    train_dataloader = DataLoader(train_sam_ds, batch_size=CFG.TRAIN_BATCH_SIZE, shuffle=False)

    model.train()

    epoch_losses = []
    epoch_ious = []

    for epoch in range(CFG.EPOCH):
        print(f'EPOCH: {epoch}')
        batch_losses = []
        batch_ious = []

        for batch in tqdm(train_dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(CFG.DEVICE),
                            input_boxes=batch["input_boxes"].to(CFG.DEVICE),
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(CFG.DEVICE)

            sam_masks_prob = torch.sigmoid(predicted_masks)
            sam_masks_prob = sam_masks_prob.squeeze()
            sam_masks = (sam_masks_prob > 0.5)

            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            batch_losses.append(loss.item())

            ious = compute_iou(sam_masks.unsqueeze(1),
                               ground_truth_masks.unsqueeze(1), ignore_empty=False)
            batch_ious.append(ious.mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_loss = mean(batch_losses)
        epoch_losses.append(mean_loss)
        print(f'Mean loss: {mean_loss}')

        mean_iou = mean([t.cpu().item() for t in batch_ious])
        epoch_ious.append(mean_iou)
        print(f'Mean IoU: {mean_iou}')

    return model, epoch_losses, epoch_ious

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_data()
    model, losses, ious = train(train_dataset, test_dataset)
    # You can add code here to save the model, plot losses and IoUs, etc.
