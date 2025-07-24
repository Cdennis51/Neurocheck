import tensorflow as tf
import numpy as np
from datasets import load_dataset

dataset = load_dataset("Falah/Alzheimer_MRI")
label_names = dataset['train'].features['label'].names
# Target size for images useful for CNN
IMG_SIZE = (224, 224)
# Images per batch for training
BATCH_SIZE = 32


def preprocess_image(img):
    # Convert grayscale to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img.astype(np.float32)

def hf_to_tf_dataset(hf_dataset, label_names, batch_size=BATCH_SIZE):
    def generator():  # Pairs image, label one at a time
        for example in hf_dataset:
            yield preprocess_image(example["image"]), example["label"]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

    # One-hot encode labels
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=len(label_names))))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_train_test_datasets(dataset, label_names, batch_size=BATCH_SIZE):

   # Prepare both train and test datasets from the Hugging Face dataset

    train_ds = hf_to_tf_dataset(dataset["train"], label_names, batch_size)
    test_ds  = hf_to_tf_dataset(dataset["test"], label_names, batch_size)
    return train_ds, test_ds
