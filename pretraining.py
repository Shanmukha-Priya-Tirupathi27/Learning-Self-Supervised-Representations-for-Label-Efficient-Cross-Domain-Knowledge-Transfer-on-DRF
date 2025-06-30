from tqdm import tqdm
from define_simclr import simclr_model, optimizer, criterion
from contrastive_loss import cont_loss
from config import DEVICE
import torch
import time
from downstream_dataloader import train_dl

EPOCHS = 101
checkpoint = 1

print("Torch-Version", torch.__version__)
print("DEVICE:", DEVICE)

for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_dl)):
        # Validate batch structure
        view1_list, view2_list = [], []
        
        for idx, v in enumerate(batch):
            if (
                isinstance(v, (list, tuple)) and len(v) == 2 and
                isinstance(v[0], torch.Tensor) and isinstance(v[1], torch.Tensor) and
                v[0].ndim == 3 and v[1].ndim == 3
            ):
                view1_list.append(v[0])
                view2_list.append(v[1])
            else:
                print(f"Skipping bad sample at index {idx}: {type(v)} | {v}")

        # Skip if nothing is valid
        if len(view1_list) == 0 or len(view2_list) == 0:
            continue

        try:
            view1_batch = torch.stack(view1_list).to(DEVICE)
            view2_batch = torch.stack(view2_list).to(DEVICE)
        except Exception as e:
            print("Skipping batch due to stack error:", e)
            continue

        # Combine views for contrastive learning
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

    # Adjust learning rate (if using scheduler, define `scheduler` in define_simclr)
    # scheduler.step()

    # Logging and saving
    if epoch % 10 == 0:
        print(f'EPOCH: {epoch+1} BATCH: {i+1} LOSS: {(running_loss/100):.4f}')
        torch.save(simclr_model.state_dict(), f'simclr_resnet50_pre_two_stage_{checkpoint}.pth')
        checkpoint += 1

    print(f'Time taken for epoch {epoch}: {((time.time() - t0) / 60):.2f} mins\n')
