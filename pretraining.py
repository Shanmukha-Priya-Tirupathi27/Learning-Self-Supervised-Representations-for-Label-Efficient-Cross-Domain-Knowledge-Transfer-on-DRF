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
        # Each batch is a list of tuples: [(view1, view2), (view1, view2), ...]
        view1_list, view2_list = [], []

        for view1, view2 in batch:
            if isinstance(view1, torch.Tensor) and isinstance(view2, torch.Tensor):
                view1_list.append(view1.unsqueeze(0))
                view2_list.append(view2.unsqueeze(0))

        if len(view1_list) == 0 or len(view2_list) == 0:
            raise ValueError("Empty batch or invalid image format.")

        # Combine into batch tensors
        view1_tensor = torch.cat(view1_list, dim=0).to(DEVICE)  # [B, C, H, W]
        view2_tensor = torch.cat(view2_list, dim=0).to(DEVICE)  # [B, C, H, W]

        # Merge both views: [2B, C, H, W]
        inputs = torch.cat([view1_tensor, view2_tensor], dim=0)

        # Forward pass
        projections = simclr_model(inputs)
        logits, labels = cont_loss(projections, temp=0.5)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Save model every 10 epochs
    if epoch % 10 == 0:
        print(f'EPOCH: {epoch+1} | LOSS: {(running_loss / 100):.4f}')
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_two_stage_{checkpoint}.pth')
        checkpoint += 1

    print(f'Time taken: {(time.time() - t0) / 60:.2f} mins')
