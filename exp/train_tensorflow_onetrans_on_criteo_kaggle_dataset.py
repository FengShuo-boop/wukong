import tensorflow as tf
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os

from model.tensorflow.onetrans import OneTrans
from model.tensorflow.lr_schedule import LinearWarmup
from data.tensorflow.criteo_kaggle_dataset import get_dataset


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
os.makedirs(f"logs/tensorflow/{formatted_time}", exist_ok=True)
file_handler = logging.FileHandler(
    f"logs/tensorflow/{formatted_time}/training.log", mode="a", encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger = logging.getLogger("wukong_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

LOGGER_PRINT_INTERVAL = 10
# TensorFlow TensorBoard Writer
summary_writer = tf.summary.create_file_writer(
    f"logs/tensorflow/{formatted_time}/tensorboard"
)

os.system("cp " + __file__ + f" logs/tensorflow/{formatted_time}/")
checkpoint_dir = f"logs/tensorflow/{formatted_time}/checkpoints"
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS:
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

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = 3
LS = 26
LNS = 13
DIM_EMB = 128
NUM_HEADS = (
    16  # number of attention heads in the token mixer（H in the paper）H must same as T
)
D_FF = 512
NUM_HIDDEN_HEAD = 2
DIM_HIDDEN_HEAD = 256
DROPOUT = 0.5
BIAS = True
GRAD_MAX_NORM = 1.0

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = OneTrans(
    num_layers=NUM_LAYERS,
    LS=LS,
    LNS=LNS,
    dim_emb=DIM_EMB,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_sparse_embs=NUM_SPARSE_EMBS,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 4096
TRAIN_EPOCHS = 10
LEARNING_RATE = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################
train_dataset = get_dataset(
    npz_file_path=NPZ_FILE_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
)
valid_dataset = get_dataset(
    npz_file_path=NPZ_FILE_PATH,
    split="valid",
    batch_size=BATCH_SIZE,
    shuffle=False,
)
logger.info(
    "Successfully loaded training and validation datasets from " + NPZ_FILE_PATH
)

####################################################################################################
#                                    BUILD MODEL WITH DUMMY INPUT                                  #
####################################################################################################
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_ = model((dummy_sparse, dummy_dense))


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataset):
    num_samples = 0
    total_logloss = 0.0
    num_batches = 0
    auc_metric = tf.keras.metrics.AUC(from_logits=True, name="val_auc")

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)

        # Calculate batch Logloss
        batch_loss = criterion(labels, outputs)
        total_logloss += batch_loss.numpy()
        num_batches += 1

        # Update AUC metric
        auc_metric.update_state(labels, outputs)
        num_samples += labels.shape[0]

    # Compute final metrics
    avg_logloss = total_logloss / num_batches if num_batches > 0 else 0.0
    auc_score = auc_metric.result().numpy()
    auc_metric.reset_states()
    return avg_logloss, auc_score, num_samples


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))
        if model.losses:
            loss += tf.add_n(model.losses)
    # Compute gradients and update parameters
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
global_step = 0
total_steps_per_epoch = 39291958 // BATCH_SIZE

for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Starting Epoch {epoch+1}/{TRAIN_EPOCHS}")
    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        labels = tf.cast(labels, tf.float32)
        train_loss = train_step(inputs, labels)

        # Log training progress
        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{total_steps_per_epoch}], "
                f"Train Loss: {train_loss.numpy():.4f}"
            )
        # Write TensorBoard summary
        with summary_writer.as_default():
            tf.summary.scalar("training_loss", train_loss, step=global_step)
        global_step += 1

    # Validation after each epoch
    val_logloss, val_auc, val_samples = validate(model, valid_dataset)
    logger.info(
        f"Validation after Epoch {epoch+1}: "
        f"Val Logloss: {val_logloss:.4f}, "
        f"Val AUC: {val_auc:.4f}, "
        f"Total Val Samples: {val_samples}"
    )
    # Write validation metrics to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar("validation_logloss", val_logloss, step=epoch + 1)
        tf.summary.scalar("validation_auc", val_auc, step=epoch + 1)

    # Save checkpoint if enabled
    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"deepfm_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        logger.info(f"Model checkpoint saved for epoch {epoch+1} at {ckpt_path}")
