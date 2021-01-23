import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from model import get_backbone, RetinaNet,DecodePredictions
from dataloader import FaceMask, resize_and_pad_image
from glob import glob

weights_dir = "retinanet"
classes_name = ['face','mask']
resnet50_backbone = get_backbone()
model = RetinaNet(2,resnet50_backbone)

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 1, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 1},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

def prepare_image(image):
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0)

val_img_paths = sorted(glob('./facedataset/images/val/*.jpg'))
val_labels_paths = sorted(glob('./facedataset/labels/val/*.txt'))

val_gen =FaceMask(val_img_paths, val_labels_paths)


a = np.random.randint(0,1800,size=5)
for i in a:
    image = val_gen[i][0]
    input_image = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        classes_name[int(x)] for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections],
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
