import torch
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.pytorch.tokenmixerlarge import TokenMixerLarge
from data.pytorch.criteo_kaggle_dataset import CriteoDataset

####################################################################################################
#                                       DDP INITIALIZATION                                         #
####################################################################################################
is_distributed = "WORLD_SIZE" in os.environ
if is_distributed:
    dist.init_process_group(backend="mccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
else:
    # for single GPU training
    local_rank = 0
    rank = 0
    world_size = 1

####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
logger = logging.getLogger("wukong_training")
logger.setLevel(logging.INFO)

if rank == 0:
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs(f"logs/pytorch/{formatted_time}", exist_ok=True)
    file_handler = logging.FileHandler(
        f"logs/pytorch/{formatted_time}/training.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    writer = SummaryWriter(log_dir=f"logs/pytorch/{formatted_time}/tensorboard")
    os.system("cp " + __file__ + f" logs/pytorch/{formatted_time}/")
    checkpoint_dir = f"logs/pytorch/{formatted_time}/checkpoints"
else:
    logger.addHandler(logging.NullHandler())
    writer = None
    checkpoint_dir = ""

LOGGER_PRINT_INTERVAL = 10
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS and rank == 0:
    os.makedirs(checkpoint_dir, exist_ok=True)

####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
NPZ_FILE_PATH = "/data/Datasets/criteo-kaggle/kaggleAdDisplayChallenge_processed.npz"
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [
    1460,
    583,
    10131227,
    2202608,
    305,
    24,
    12517,
    633,
    3,
    93145,
    5683,
    8351593,
    3194,
    27,
    14992,
    5461306,
    10,
    5652,
    2173,
    4,
    7046547,
    18,
    15,
    286181,
    105,
    142572,
]
DIM_OUTPUT = 1

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = 6
DIM_EMB = 128
NUM_HEADS = (
    16  # number of attention heads in the token mixer（H in the paper）H must same as T
)
NUM_EXPERTS = 8
TOP_K = 4
NUM_HIDDEN_HEAD = 2
DIM_HIDDEN_HEAD = 256
DROPOUT = 0.5
BIAS = True
DTYPE = torch.float32

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
DEVICE_STR = "musa"
if is_distributed:
    DEVICE = torch.device(f"{DEVICE_STR}:{local_rank}")
else:
    DEVICE = torch.device(DEVICE_STR)

BATCH_SIZE = 16384 // world_size
TRAIN_EPOCHS = 10
PEAK_LR = 0.004
INIT_LR = 1e-8
GRAD_MAX_NORM = 1.0

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = TokenMixerLarge(
    group_dims=[[DIM_EMB] * NUM_CAT_FEATURES, [DIM_EMB] * NUM_DENSE_FEATURES],
    num_layers=NUM_LAYERS,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    dim_emb=DIM_EMB,
    num_heads=NUM_HEADS,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
    dim_output=DIM_OUTPUT,
    dropout=DROPOUT,
    bias=BIAS,
)

model.to(DEVICE)

if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

####################################################################################################
#                                     OPTIMIZER & SCHEDULER                                        #
####################################################################################################
critrion = torch.nn.BCEWithLogitsLoss()
raw_model = model.module if is_distributed else model
embedding_parameters = [
    param
    for name, param in raw_model.named_parameters()
    if "embedding.sparse_embedding" in name
]
other_parameters = [
    param
    for name, param in raw_model.named_parameters()
    if "embedding.sparse_embedding" not in name
]
embedding_optimizer = torch.optim.SGD(embedding_parameters, lr=PEAK_LR)
other_optimizer = torch.optim.Adam(other_parameters, lr=PEAK_LR)

total_iters_estimate = 39291958 // (BATCH_SIZE * world_size)
embedding_optimizer_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    embedding_optimizer,
    start_factor=INIT_LR / PEAK_LR,
    total_iters=total_iters_estimate,
)
other_optimizer_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    other_optimizer,
    start_factor=INIT_LR / PEAK_LR,
    total_iters=total_iters_estimate,
)

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################
train_dataset = CriteoDataset(NPZ_FILE_PATH, split="train")
valid_dataset = CriteoDataset(NPZ_FILE_PATH, split="valid")
train_sampler = (
    DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
)
valid_sampler = (
    DistributedSampler(valid_dataset, shuffle=False) if is_distributed else None
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    sampler=train_sampler,
)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    sampler=valid_sampler,
)


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataloader):
    model.eval()
    num_samples = torch.tensor(0.0, device=DEVICE)
    num_correct = torch.tensor(0.0, device=DEVICE)
    pos_samples = torch.tensor(0.0, device=DEVICE)
    pos_correct = torch.tensor(0.0, device=DEVICE)

    with torch.no_grad():
        for sparse_inputs, dense_inputs, labels in dataloader:
            sparse_inputs = sparse_inputs.to(DEVICE)
            dense_inputs = dense_inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sparse_inputs, dense_inputs)
            predictions = outputs.squeeze() >= 0

            num_samples += labels.size(0)
            pos_samples += (labels == 1).sum()
            pos_correct += ((predictions == 1) & (labels == 1)).sum()
            num_correct += (predictions == labels).sum()

    if is_distributed:
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_correct, op=dist.ReduceOp.SUM)

    model.train()

    num_samples_val = num_samples.item()
    pos_samples_val = pos_samples.item()

    accuracy = num_correct.item() / num_samples_val if num_samples_val > 0 else 0
    recall_pos = pos_correct.item() / pos_samples_val if pos_samples_val > 0 else 0

    return accuracy, num_samples_val, recall_pos, pos_samples_val


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
scaler = torch.amp.GradScaler(DEVICE_STR, enabled=(torch.bfloat16 == DTYPE))
model.train()

if rank == 0:
    logger.info("Use device: " + DEVICE_STR)
    logger.info("Use dtype: " + str(DTYPE))
    logger.info(f"Distributed: {is_distributed}, World Size: {world_size}")

step = 0
for epoch in range(TRAIN_EPOCHS):
    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    for batch_idx, (sparse_inputs, dense_inputs, labels) in enumerate(train_dataloader):
        sparse_inputs = sparse_inputs.to(DEVICE)
        dense_inputs = dense_inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if torch.bfloat16 == DTYPE:
            with torch.amp.autocast(DEVICE_STR):
                outputs = model(
                    sparse_inputs,
                    dense_inputs.to(torch.bfloat16),
                )
                loss = critrion(
                    outputs.squeeze(),
                    labels.to(torch.bfloat16),
                )

            model.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
            scaler.step(embedding_optimizer)
            scaler.step(other_optimizer)
            scaler.update()
            embedding_optimizer_lr_scheduler.step()
            other_optimizer_lr_scheduler.step()

        elif torch.float32 == DTYPE:
            outputs = model(sparse_inputs, dense_inputs)
            loss = critrion(outputs.squeeze(), labels.to(torch.float32))
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
            embedding_optimizer.step()
            other_optimizer.step()
            embedding_optimizer_lr_scheduler.step()
            other_optimizer_lr_scheduler.step()
        else:
            raise ValueError("Unsupported DTYPE. Use torch.bfloat16 or torch.float32.")

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0 and rank == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Embedding LR: {embedding_optimizer_lr_scheduler.get_last_lr()[0]:.6f}, "
                f"Other LR: {other_optimizer_lr_scheduler.get_last_lr()[0]:.6f}"
            )

        if writer is not None:
            writer.add_scalar("training_loss", loss, step)
            writer.add_scalar(
                "embedding_optimizer_lr",
                embedding_optimizer_lr_scheduler.get_last_lr()[0],
                step,
            )
            writer.add_scalar(
                "other_optimizer_lr",
                other_optimizer_lr_scheduler.get_last_lr()[0],
                step,
            )
        step += 1

    accuracy, num_samples, recall_pos, pos_samples = validate(model, valid_dataloader)

    if rank == 0:
        logger.info(
            f"Validation after Epoch {epoch+1}: "
            f"Accuracy: {accuracy*100:.2f}%, "
            f"Total Samples: {num_samples}, "
            f"Positive Recall: {recall_pos*100:.2f}%, "
            f"Positive Samples: {pos_samples}"
        )
        if writer is not None:
            writer.add_scalar("validation_accuracy", accuracy, epoch + 1)
            writer.add_scalar("validation_recall_pos", recall_pos, epoch + 1)

        if SAVE_CHECKPOINTS:
            state_dict = (
                model.module.state_dict() if is_distributed else model.state_dict()
            )
            torch.save(
                state_dict,
                os.path.join(checkpoint_dir, f"wukong_epoch_{epoch+1}.pth"),
            )
            logger.info(
                f"Model checkpoint saved for epoch {epoch+1} at {checkpoint_dir}/wukong_epoch_{epoch+1}.pth"
            )

if is_distributed:
    dist.destroy_process_group()
