import tensorflow as tf
tf.load_library("/workspace/tensorflow_musa_extension/build/libmusa_plugin.so")
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os
from model.tensorflow.xdeepfm import XDeepFM
from model.tensorflow.lr_schedule import LinearWarmup
from data.tensorflow.criteo_kaggle_dataset import get_dataset

# Reproducibility settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Logger configuration
now = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
os.makedirs(f"logs/tensorflow/{now}", exist_ok=True)

file_handler = logging.FileHandler(f"logs/tensorflow/{now}/training.log", encoding="utf-8")
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)

logger = logging.getLogger("xdeepfm")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
LOGGER_PRINT_INTERVAL = 10

# TensorBoard writer
summary_writer = tf.summary.create_file_writer(f"logs/tensorflow/{now}/tensorboard")

# Checkpoint settings
checkpoint_dir = f"logs/tensorflow/{now}/checkpoints"
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset parameters (Criteo Kaggle)
NPZ_FILE_PATH = "/data/Datasets/criteo-kaggle/kaggleAdDisplayChallenge_processed.npz"
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]
DIM_OUTPUT = 1

# Model hyperparameters
DIM_EMB = 16
CIN_LAYER_SIZE = (128, 128)
CIN_SPLIT_HALF = True
CIN_ACTIVATION = None
L2_REG_LINEAR = 1e-6
L2_REG_CIN = 1e-6
L2_REG_DNN = 1e-6
DNN_HIDDEN_UNITS = (256, 64)
DROPOUT = 0
BIAS = False

# Training parameters
BATCH_SIZE = 16384
TRAIN_EPOCHS = 10
PEAK_LR = 0.001
INIT_LR = 1e-5
TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH

# Model initialization
model = XDeepFM(
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    cin_layer_size=CIN_LAYER_SIZE,
    cin_split_half=CIN_SPLIT_HALF,
    cin_activation=CIN_ACTIVATION,
    l2_reg_linear=L2_REG_LINEAR,
    l2_reg_cin=L2_REG_CIN,
    l2_reg_dnn=L2_REG_DNN,
    dnn_hidden_units=DNN_HIDDEN_UNITS,
    dnn_dropout=DROPOUT,
    dim_output=DIM_OUTPUT,
    bias=BIAS,
    seed=SEED
)

# Learning rate scheduler
lr_schedule = LinearWarmup(INIT_LR, PEAK_LR, TOTAL_ITERS)

# Optimizers
embedding_optimizer = tf.keras.optimizers.SGD(lr_schedule)
other_optimizer = tf.keras.optimizers.Adam(lr_schedule)

# Loss function
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Dataset loader
train_dataset = get_dataset(NPZ_FILE_PATH, "train", BATCH_SIZE, shuffle=True)
valid_dataset = get_dataset(NPZ_FILE_PATH, "valid", BATCH_SIZE, shuffle=False)

# Build model with dummy input
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), tf.float32)
_ = model((dummy_sparse, dummy_dense))

# Separate trainable parameters
embedding_params = []
other_params = []
for var in model.trainable_variables:
    if "sparse_embedding" in var.name:
        embedding_params.append(var)
    else:
        other_params.append(var)

# Validation function
def validate(model, dataset):
    num_samples = num_correct = pos_samples = pos_correct = 0
    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        labels = tf.cast(labels, tf.float32)
        preds = tf.cast(tf.squeeze(outputs) >= 0, tf.float32)
        
        num_samples += labels.shape[0]
        pos_samples += tf.reduce_sum(labels).numpy()
        num_correct += tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()
        pos_correct += tf.reduce_sum(tf.boolean_mask(preds, tf.equal(labels, 1.0))).numpy()
    
    accuracy = num_correct / num_samples if num_samples else 0.0
    recall = pos_correct / pos_samples if pos_samples else 0.0
    return accuracy, num_samples, recall, pos_samples

# Training step
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))
    
    grads = tape.gradient(loss, model.trainable_variables)
    emb_grads, other_grads = [], []
    for g, v in zip(grads, model.trainable_variables):
        if g is None: continue
        if "sparse_embedding" in v.name:
            emb_grads.append((g, v))
        else:
            other_grads.append((g, v))
    
    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)
    return loss

# Training loop
step = 0
for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Epoch {epoch+1}/{TRAIN_EPOCHS} Started")
    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        labels = tf.cast(labels, tf.float32)
        loss = train_step(inputs, labels)
        lr = lr_schedule(step)

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.numpy():.4f}, LR: {lr.numpy():.6f}")
        
        with summary_writer.as_default():
            tf.summary.scalar("train_loss", loss, step)
            tf.summary.scalar("lr", lr, step)
        step += 1

    # Validation
    acc, total, recall, pos = validate(model, valid_dataset)
    logger.info(f"Epoch {epoch+1} Val | Acc: {acc*100:.2f}% | Recall: {recall*100:.2f}% | Samples: {total}")
    
    with summary_writer.as_default():
        tf.summary.scalar("val_acc", acc, epoch+1)
        tf.summary.scalar("val_recall", recall, epoch+1)

    # Save checkpoint
    if SAVE_CHECKPOINTS:
        model.save_weights(os.path.join(checkpoint_dir, f"epoch_{epoch+1}"))