import tensorflow as tf

# tf.load_library("/workspace/tensorflow_musa_extension/build/libmusa_plugin.so")
import numpy as np
import random
import logging
import sys
from datetime import datetime
import os
from model.tensorflow.dien import DIEN
from model.tensorflow.lr_schedule import LinearWarmup

# Use existing Wukong dataset loader
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
logger = logging.getLogger("dien_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
LOGGER_PRINT_INTERVAL = 10
# TensorFlow TensorBoard Writer
summary_writer = tf.summary.create_file_writer(
    f"logs/tensorflow/{formatted_time}/tensorboard"
)
# Backup training script for reproducibility
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
# Sequence configuration (for DIEN compatibility layer)
NUM_SEQ_EMBS = NUM_SPARSE_EMBS[2]  # Use item id vocab size
MAX_SEQ_LEN = 1  # Temporarily set to 1 for compatibility
DIM_OUTPUT = 1
USE_AUXILIARY_LOSS = False  # Disable auxiliary loss temporarily
AUX_LOSS_ALPHA = 1.0

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
DIM_EMB = 32
EXTRACTOR_HIDDEN_DIM = 32
EVOLUTION_HIDDEN_DIM = 32
GRU_TYPE = "AUGRU"
ATT_HIDDEN_UNITS = (64, 16)
ATT_ACTIVATION = "dice"
ATT_WEIGHT_NORMALIZATION = False
NUM_HIDDEN_HEAD = 2
DIM_HIDDEN_HEAD = 64
DROPOUT = 0.3
BIAS = False

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = DIEN(
    num_sparse_embs=NUM_SPARSE_EMBS,
    num_seq_embs=NUM_SEQ_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    max_seq_len=MAX_SEQ_LEN,
    extractor_hidden_dim=EXTRACTOR_HIDDEN_DIM,
    evolution_hidden_dim=EVOLUTION_HIDDEN_DIM,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
    use_auxiliary_loss=USE_AUXILIARY_LOSS,
    gru_type=GRU_TYPE,
    att_hidden_units=ATT_HIDDEN_UNITS,
    att_activation=ATT_ACTIVATION,
    att_weight_normalization=ATT_WEIGHT_NORMALIZATION,
    dim_output=DIM_OUTPUT,
    dropout=DROPOUT,
    bias=BIAS,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 128
TRAIN_EPOCHS = 10
PEAK_LR = 0.001
INIT_LR = 1e-8
TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH * TRAIN_EPOCHS
lr_schedule = LinearWarmup(
    initial_learning_rate=INIT_LR,
    peak_learning_rate=PEAK_LR,
    warmup_steps=TOTAL_STEPS_PER_EPOCH,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
#                         ADAPTER: WRAP DATASET FOR DIEN INPUT FORMAT                            #
####################################################################################################
def wrap_dataset_for_dien(dataset):
    """
    Wrap Wukong dataset format to DIEN input format
    Wukong inputs: (sparse_inputs, dense_inputs)
    DIEN inputs: (sparse_inputs, dense_inputs, seq_inputs, seq_length)
    """

    def adapter(inputs, labels):
        sparse_inputs, dense_inputs = inputs
        batch_size = tf.shape(sparse_inputs)[0]

        # Create dummy sequence inputs (use item id feature as sequence)
        # In real DIEN, this should be a history of item ids
        seq_inputs = sparse_inputs[:, 2:3]  # Use item id column as dummy sequence
        seq_length = tf.ones((batch_size, 1), dtype=tf.int32)

        dien_inputs = (sparse_inputs, dense_inputs, seq_inputs, seq_length)
        return dien_inputs, labels

    return dataset.map(adapter)


# Apply adapter
train_dataset = wrap_dataset_for_dien(train_dataset)
valid_dataset = wrap_dataset_for_dien(valid_dataset)

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
# Trigger model build with dummy inputs
dummy_static_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
dummy_seq = tf.zeros((1, MAX_SEQ_LEN), dtype=tf.int32)
dummy_seq_len = tf.constant([[MAX_SEQ_LEN]], dtype=tf.int32)
dummy_inputs = (dummy_static_sparse, dummy_dense, dummy_seq, dummy_seq_len)
_ = model(dummy_inputs, training=False)

# Separate embedding and other parameters
embedding_parameters = []
other_parameters = []
for var in model.trainable_variables:
    var_identifier = var.path if hasattr(var, "path") else var.name
    if "sparse_embedding" in var_identifier and "embeddings" in var.name:
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
    total_loss = 0.0
    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)
        batch_loss = criterion(labels, outputs)
        total_loss += batch_loss.numpy() * labels.shape[0]
        predictions = tf.cast(outputs >= 0, tf.float32)
        num_samples += labels.shape[0]
        pos_samples += tf.reduce_sum(labels).numpy()
        correct_preds = tf.cast(tf.equal(predictions, labels), tf.float32)
        num_correct += tf.reduce_sum(correct_preds).numpy()
        pos_mask = tf.equal(labels, 1.0)
        pos_correct += tf.reduce_sum(tf.boolean_mask(predictions, pos_mask)).numpy()
    accuracy = num_correct / num_samples if num_samples > 0 else 0
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0
    return accuracy, avg_loss, num_samples, recall_pos, pos_samples


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        main_loss = criterion(labels, tf.squeeze(outputs))
        total_loss = main_loss
        if USE_AUXILIARY_LOSS and model.losses:
            aux_loss = tf.add_n(model.losses)
            total_loss += AUX_LOSS_ALPHA * aux_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, main_loss, (aux_loss if USE_AUXILIARY_LOSS else 0.0)


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
global_step = 0
for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Starting Epoch {epoch+1}/{TRAIN_EPOCHS}")
    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        labels = tf.cast(labels, tf.float32)
        total_loss, main_loss, aux_loss = train_step(inputs, labels)
        current_lr = lr_schedule(global_step)

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            log_msg = (
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{TOTAL_STEPS_PER_EPOCH}], "
                f"Total Loss: {total_loss.numpy():.4f}, "
                f"Main Loss: {main_loss.numpy():.4f}, "
            )
            if USE_AUXILIARY_LOSS:
                log_msg += f"Aux Loss: {aux_loss.numpy():.4f}, "
            log_msg += f"LR: {current_lr.numpy():.8f}"
            logger.info(log_msg)

        with summary_writer.as_default():
            tf.summary.scalar("train_total_loss", total_loss, step=global_step)
            tf.summary.scalar("train_main_loss", main_loss, step=global_step)
            tf.summary.scalar("optimizer_lr", current_lr, step=global_step)

        global_step += 1

    # Validation
    val_accuracy, val_loss, val_num_samples, val_recall_pos, val_pos_samples = validate(
        model, valid_dataset
    )
    logger.info(
        f"Validation after Epoch {epoch+1}: "
        f"Val Loss: {val_loss:.4f}, "
        f"Accuracy: {val_accuracy*100:.2f}%, "
        f"Total Samples: {val_num_samples}, "
        f"Positive Recall: {val_recall_pos*100:.2f}%, "
        f"Positive Samples: {val_pos_samples}"
    )
    with summary_writer.as_default():
        tf.summary.scalar("validation_loss", val_loss, step=epoch + 1)
        tf.summary.scalar("validation_accuracy", val_accuracy, step=epoch + 1)
        tf.summary.scalar("validation_recall_pos", val_recall_pos, step=epoch + 1)

    # Checkpoint saving
    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"dien_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        logger.info(f"Model checkpoint saved for epoch {epoch+1} at {ckpt_path}")
