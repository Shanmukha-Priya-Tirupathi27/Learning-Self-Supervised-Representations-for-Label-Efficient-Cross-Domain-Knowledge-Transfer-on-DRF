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

    for i, batch in enumerate(tqdm(train_dl)):
        # Unpack the two views and move them to DEVICE
        view1_batch = torch.stack([v[0] for v in batch]).to(DEVICE)
        view2_batch = torch.stack([v[1] for v in batch]).to(DEVICE)
        inputs = torch.cat([view1_batch, view2_batch], dim=0)

        # Forward pass
        projections = simclr_model(inputs)
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 10 == 0:
        print(f'EPOCH: {epoch+1} | LOSS: {(running_loss / 100):.4f}')
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_two_stage_{checkpoint}.pth')
        checkpoint += 1

    print(f'Time taken: {(time.time() - t0)/60:.2f} mins')

