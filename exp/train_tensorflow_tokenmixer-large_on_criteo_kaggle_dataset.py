import tensorflow as tf
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os

from model.tensorflow.tokenmixerlarge import TokenMixerLarge
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

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 16384
TRAIN_EPOCHS = 10
PEAK_LR = 0.004
INIT_LR = 1e-8
TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH

lr_schedule = LinearWarmup(
    initial_learning_rate=INIT_LR, peak_learning_rate=PEAK_LR, warmup_steps=TOTAL_ITERS
)
embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
# TF Lazy Execution
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_ = model((dummy_sparse, dummy_dense))

embedding_parameters = []
other_parameters = []

for var in model.trainable_variables:
    if hasattr(var, "path"):
        # path is available in TF 2.13+
        if "sparse_embedding" in var.path and "embeddings" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)
    else:
        if "sparse_embedding" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)

logger.info(f"Number of embedding parameters: {len(embedding_parameters)}")
logger.info(f"Number of other parameters: {len(other_parameters)}")


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataset):
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)

        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)

        predictions = tf.cast(outputs >= 0, tf.float32)

        num_samples += labels.shape[0]
        pos_samples += tf.reduce_sum(labels).numpy()

        correct_preds = tf.cast(tf.equal(predictions, labels), tf.float32)
        num_correct += tf.reduce_sum(correct_preds).numpy()

        pos_mask = tf.equal(labels, 1.0)
        pos_correct += tf.reduce_sum(tf.boolean_mask(predictions, pos_mask)).numpy()

    accuracy = num_correct / num_samples if num_samples > 0 else 0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0
    return accuracy, num_samples, recall_pos, pos_samples


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)
    emb_grads = []
    other_grads = []

    # 梯度裁剪
    clipped_grads = []
    for grad in grads:
        if grad is not None:
            clipped_grad = tf.clip_by_norm(grad, GRAD_MAX_NORM)  # 在这里裁剪梯度
            clipped_grads.append(clipped_grad)
        else:
            clipped_grads.append(None)

    for grad, var in zip(clipped_grads, model.trainable_variables):
        if grad is not None:
            if hasattr(var, "path"):
                # path is available in TF 2.13+
                if "sparse_embedding" in var.path and "embeddings" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
            else:
                if "sparse_embedding" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))

    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
step = 0
embedding_lr_metric = tf.keras.metrics.Mean()
other_lr_metric = tf.keras.metrics.Mean()

for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Starting Epoch {epoch+1}/{TRAIN_EPOCHS}")

    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        labels = tf.cast(labels, tf.float32)

        loss = train_step(inputs, labels)
        current_lr = lr_schedule(step)

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{TOTAL_STEPS_PER_EPOCH}], "
                f"Loss: {loss.numpy():.4f}, "
                f"LR: {current_lr.numpy():.6f}"
            )

        with summary_writer.as_default():
            tf.summary.scalar("training_loss", loss, step=step)
            tf.summary.scalar("optimizer_lr", current_lr, step=step)

        step += 1

    accuracy, num_samples, recall_pos, pos_samples = validate(model, valid_dataset)

    logger.info(
        f"Validation after Epoch {epoch+1}: "
        f"Accuracy: {accuracy*100:.2f}%, "
        f"Total Samples: {num_samples}, "
        f"Positive Recall: {recall_pos*100:.2f}%, "
        f"Positive Samples: {pos_samples}"
    )

    with summary_writer.as_default():
        tf.summary.scalar("validation_accuracy", accuracy, step=epoch + 1)
        tf.summary.scalar("validation_recall_pos", recall_pos, step=epoch + 1)

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"wukong_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        logger.info(f"Model checkpoint saved for epoch {epoch+1} at {ckpt_path}")
