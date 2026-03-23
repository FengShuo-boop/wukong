import tensorflow as tf
tf.load_library("/workspace/tensorflow_musa_extension/build/libmusa_plugin.so")

import numpy as np
import random
import logging
import sys
from datetime import datetime
import os

from model.tensorflow.din import DIN
from model.tensorflow.lr_schedule import LinearWarmup
from data.tensorflow.criteo_kaggle_dataset import get_dataset


####################################################################################################
#                                        SET RANDOM SEEDS                                          #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
now            = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")
formatter      = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.makedirs(f"logs/tensorflow_din/{formatted_time}", exist_ok=True)

file_handler = logging.FileHandler(
    f"logs/tensorflow_din/{formatted_time}/training.log", mode="a", encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger = logging.getLogger("din_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

LOGGER_PRINT_INTERVAL = 10

summary_writer = tf.summary.create_file_writer(
    f"logs/tensorflow_din/{formatted_time}/tensorboard"
)

os.system("cp " + __file__ + f" logs/tensorflow_din/{formatted_time}/")

checkpoint_dir  = f"logs/tensorflow_din/{formatted_time}/checkpoints"
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)


####################################################################################################
#                               DATASET SPECIFIC CONFIGURATION                                     #
####################################################################################################
NPZ_FILE_PATH    = "/data/Datasets/criteo-kaggle/kaggleAdDisplayChallenge_processed.npz"
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS  = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3,
    93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652,
    2173, 4, 7046547, 18, 15, 286181, 105, 142572,
]
DIM_OUTPUT = 1


####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
# ── Embedding ────────────────────────────────────────────────────────────────────────────────────
DIM_EMB = 64            # Embedding dimension D

NUM_TARGET_FEATURES = 3

# ── Local activation unit (attention MLP) ─────────────────────────────────────────────────────────
ATT_HIDDEN_UNITS  = (80, 40)
ATT_ACTIVATION    = 'dice'   

# ── Final DNN ─────────────────────────────────────────────────────────────────────────────────────
DNN_HIDDEN_UNITS  = (256, 128, 64)
DNN_ACTIVATION    = 'dice'   # 'dice' | 'prelu' | 'relu'
DNN_USE_BN        = False    # False is the default in the reference implementation
DROPOUT           = 0.5      # consistent with Wukong
BIAS              = False    # consistent with Wukong


####################################################################################################
#                                          CREATE MODEL                                            #
####################################################################################################
model = DIN(
    num_sparse_embs    = NUM_SPARSE_EMBS,
    dim_emb            = DIM_EMB,
    dim_input_sparse   = NUM_CAT_FEATURES,
    dim_input_dense    = NUM_DENSE_FEATURES,
    num_target_features= NUM_TARGET_FEATURES,
    att_hidden_units   = ATT_HIDDEN_UNITS,
    att_activation     = ATT_ACTIVATION,
    dnn_hidden_units   = DNN_HIDDEN_UNITS,
    dnn_activation     = DNN_ACTIVATION,
    dnn_use_bn         = DNN_USE_BN,
    dropout            = DROPOUT,
    bias               = BIAS,
)


####################################################################################################
#                                 TRAINING SPECIFIC CONFIGURATION                                  #
####################################################################################################
BATCH_SIZE             = 16384
TRAIN_EPOCHS           = 10

PEAK_LR                = 0.001
INIT_LR                = 1e-8
TOTAL_STEPS_PER_EPOCH  = 39291958 // BATCH_SIZE   # ≈ 2399 steps/epoch
TOTAL_ITERS            = TOTAL_STEPS_PER_EPOCH    # warmup over first epoch

lr_schedule = LinearWarmup(
    initial_learning_rate = INIT_LR,
    peak_learning_rate    = PEAK_LR,
    warmup_steps          = TOTAL_ITERS,
)

embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
other_optimizer     = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)


####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################
train_dataset = get_dataset(
    npz_file_path = NPZ_FILE_PATH,
    split         = "train",
    batch_size    = BATCH_SIZE,
    shuffle       = True,
)
valid_dataset = get_dataset(
    npz_file_path = NPZ_FILE_PATH,
    split         = "valid",
    batch_size    = BATCH_SIZE,
    shuffle       = False,
)


####################################################################################################
#                                  BUILD MODEL & SEPARATE VARS                                     #
####################################################################################################
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES),   dtype=tf.int32)
dummy_dense  = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_            = model((dummy_sparse, dummy_dense))

embedding_parameters = []
other_parameters     = []

for var in model.trainable_variables:

    var_id = var.path if hasattr(var, "path") else var.name
    if "sparse_embedding" in var_id and "embeddings" in var.name:
        embedding_parameters.append(var)
    else:
        other_parameters.append(var)

logger.info(f"Model: DIN  |  dim_emb={DIM_EMB}  |  "
            f"num_target={NUM_TARGET_FEATURES}  |  "
            f"att={ATT_HIDDEN_UNITS}  |  dnn={DNN_HIDDEN_UNITS}")
logger.info(f"Total trainable vars  : {len(model.trainable_variables)}")
logger.info(f"Embedding parameters  : {len(embedding_parameters)}")
logger.info(f"Other parameters      : {len(other_parameters)}")

total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
logger.info(f"Total parameter count : {total_params:,}")


####################################################################################################
#                                         VALID FUNCTION                                           #
####################################################################################################
def validate(model, dataset):

    auc_metric = tf.keras.metrics.AUC(name='auc')
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0

    for inputs, labels in dataset:
        labels  = tf.cast(labels, tf.float32)
        outputs = model(inputs, training=False)
        outputs = tf.squeeze(outputs)             # (bs,)

        # Probabilities for AUC
        probs = tf.sigmoid(outputs)
        auc_metric.update_state(labels, probs)

        predictions = tf.cast(outputs >= 0.0, tf.float32)

        num_samples += labels.shape[0]
        pos_samples += int(tf.reduce_sum(labels).numpy())

        correct     = tf.cast(tf.equal(predictions, labels), tf.float32)
        num_correct += int(tf.reduce_sum(correct).numpy())

        pos_mask    = tf.equal(labels, 1.0)
        pos_correct += int(tf.reduce_sum(tf.boolean_mask(predictions, pos_mask)).numpy())

    accuracy   = num_correct / num_samples if num_samples > 0 else 0.0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0.0
    auc        = float(auc_metric.result().numpy())

    return accuracy, num_samples, recall_pos, pos_samples, auc


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss    = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)

    emb_grads   = []
    other_grads = []

    for grad, var in zip(grads, model.trainable_variables):
        if grad is None:
            continue
        var_id = var.path if hasattr(var, "path") else var.name
        if "sparse_embedding" in var_id and "embeddings" in var.name:
            emb_grads.append((grad, var))
        else:
            other_grads.append((grad, var))

    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


####################################################################################################
#                                          TRAINING LOOP                                           #
####################################################################################################
step = 0

for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Starting Epoch {epoch + 1}/{TRAIN_EPOCHS}")

    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        labels = tf.cast(labels, tf.float32)
        loss   = train_step(inputs, labels)
        current_lr = lr_schedule(step)

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{TOTAL_STEPS_PER_EPOCH}], "
                f"Loss: {loss.numpy():.4f}, "
                f"LR: {current_lr.numpy():.6f}"
            )

        with summary_writer.as_default():
            tf.summary.scalar("training_loss", loss,       step=step)
            tf.summary.scalar("optimizer_lr",  current_lr, step=step)

        step += 1

    accuracy, num_samples, recall_pos, pos_samples, auc = validate(
        model, valid_dataset
    )

    logger.info(
        f"Validation after Epoch {epoch + 1}: "
        f"AUC: {auc:.4f}, "
        f"Accuracy: {accuracy * 100:.2f}%, "
        f"Positive Recall: {recall_pos * 100:.2f}%, "
        f"Total Samples: {num_samples}, "
        f"Positive Samples: {pos_samples}"
    )

    with summary_writer.as_default():
        tf.summary.scalar("validation_auc",          auc,        step=epoch + 1)
        tf.summary.scalar("validation_accuracy",     accuracy,   step=epoch + 1)
        tf.summary.scalar("validation_recall_pos",   recall_pos, step=epoch + 1)

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"din_epoch_{epoch + 1}")
        model.save_weights(ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")