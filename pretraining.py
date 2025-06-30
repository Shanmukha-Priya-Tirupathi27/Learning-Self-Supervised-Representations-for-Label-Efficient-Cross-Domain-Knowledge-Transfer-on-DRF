from tqdm import tqdm
import torch
import time
from define_simclr import simclr_model, optimizer, criterion
from contrastive_loss import cont_loss
from config import DEVICE
from downstream_dataloader import train_dl

EPOCHS = 101
checkpoint = 1

for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0

    for i, (view1_batch, view2_batch) in enumerate(tqdm(train_dl)):
        # Each view is a batch of images: shape [B, C, H, W]
        view1 = view1_batch.to(DEVICE)
        view2 = view2_batch.to(DEVICE)

        # Combine both views: [2B, C, H, W]
        inputs = torch.cat([view1, view2], dim=0)

        # Forward pass
        projections = simclr_model(inputs)
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 10 == 0:
        print(f'EPOCH: {epoch+1} | LOSS: {(running_loss / 100):.4f}')
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_two_stage_{checkpoint}.pth')
        checkpoint += 1

    print(f'Time taken: {(time.time() - t0) / 60:.2f} mins')
