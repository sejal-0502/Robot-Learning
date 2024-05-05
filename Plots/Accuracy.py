mport tensorflow as tf
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Load CSV data into a pandas DataFrame
csv_file = 'Accuracy.csv'
df = pd.read_csv(csv_file)

# Initialize TensorFlow SummaryWriter
log_dir = '/project/dl2024s/DL2024Lab_SejalM/Screenshots'
writer = tf.summary.create_file_writer(log_dir)

# Extract 'train' and 'validation' columns from DataFrame
train_values = df['Train_accuracy']
validation_values = df['Validation_accuracy']

# Ensure 'train' and 'validation' columns have the same length
assert len(train_values) == len(validation_values), "Train and validation data lengths must match"

with writer.as_default():
    # Write 'train' and 'validation' values as scalar metrics
    for index, (train_val, valid_val) in enumerate(zip(train_values, validation_values)):
        tf.summary.scalar('Train_accuracy', train_val, step=index)
        tf.summary.scalar('Validation_accuracy', valid_val, step=index)

    # Flush writer to write data to disk
    writer.flush()

