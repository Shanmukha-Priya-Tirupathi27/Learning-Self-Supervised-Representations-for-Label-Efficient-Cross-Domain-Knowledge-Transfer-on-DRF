from tqdm import tqdm
from define_simclr import simclr_model, optimizer, criterion
from contrastive_loss import cont_loss
from config import DEVICE
import torch
import time
from downstream_dataloader import train_dl

EPOCHS = 30
checkpoint = 1

print("Torch-Version", torch.__version__)
print("DEVICE:", DEVICE)

for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0

    for i, batch in enumerate(tqdm(train_dl)):
        view1_list, view2_list = [], []

        # Handle known broken batch formats
        if isinstance(batch, list) and len(batch) == 2 and isinstance(batch[0], list):
            # Common format: [ [view1, view2, ...], labels ]
            possible_data = batch[0]
        else:
            possible_data = batch

        # Filter proper (view1, view2) pairs
        for idx, sample in enumerate(possible_data):
            if isinstance(sample, (list, tuple)) and len(sample) == 2:
                v1, v2 = sample
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor) and v1.ndim == 3 and v2.ndim == 3:
                    view1_list.append(v1)
                    view2_list.append(v2)
                else:
                    print(f"[EPOCH {epoch}] Skipping invalid tensor shapes at index {idx}")
            else:
                print(f"[EPOCH {epoch}] Skipping malformed sample at index {idx}: {type(sample)} | {sample}")

        # If no valid views, skip batch
        if len(view1_list) == 0 or len(view2_list) == 0:
            continue

        try:
            view1_batch = torch.stack(view1_list).to(DEVICE)
            view2_batch = torch.stack(view2_list).to(DEVICE)
        except Exception as e:
            print("Stacking error:", e)
            continue

        inputs = torch.cat([view1_batch, view2_batch], dim=0)

        # Forward pass
        projections = simclr_model(inputs)
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Save checkpoint
    if epoch % 10 == 0:
        print(f'EPOCH: {epoch+1} | LOSS: {(running_loss/len(train_dl)):.4f}')
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_epoch_{checkpoint}.pth')
        checkpoint += 1

    print(f'[EPOCH {epoch+1}] Time: {(time.time() - t0)/60:.2f} mins\n')
