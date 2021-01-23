import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from glob import glob

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from model import RetinaNet,get_backbone
from loss import RetinaNetLoss 
from dataloader import FaceMask
from box import LabelEncoder


num_classes = 2
classes_name = ['face','mask']
batch_size = 2
label_encoder = LabelEncoder()


learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)


optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)



train_img_paths = sorted(glob('./facedataset/images/train/*.jpg'))
train_labels_paths = sorted(glob('./facedataset/labels/train/*.txt'))
val_img_paths = sorted(glob('./facedataset/images/val/*.jpg'))
val_labels_paths = sorted(glob('./facedataset/labels/val/*.txt'))


train_gen =FaceMask(train_img_paths, train_labels_paths)
train_dataset = tf.data.Dataset.from_generator(
     lambda: train_gen,
     output_types=(tf.float32,tf.float32,tf.int32),
     output_shapes=((640, 640, 3),(None, 4),(None,)))

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)

train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)


val_gen =FaceMask(val_img_paths, val_labels_paths)
val_dataset = tf.data.Dataset.from_generator(
     lambda: val_gen,
     output_types=(tf.float32,tf.float32,tf.int32),
     output_shapes=((640, 640, 3),(None, 4),(None,)))

val_dataset = val_dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)

val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

epochs = 30


model_dir = "retinanet/"
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    ),
    tb_callback
]

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)