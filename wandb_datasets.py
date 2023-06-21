from PIL import Image
import util
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
import wandb
from mmengine.utils import ProgressBar
# default project name where results will be logged
WANDB_PROJECT = "dsviz-demo"
wandb.login()

# set SIZE to "TINY", "MEDIUM" or (read the warning) "LARGE"
# to select one of these datasets
# TINY dataset: 50 images
# MEDIUM dataset: 200 images
# warning: you may run out of RAM in Colab & need to restart the notebook
# between steps at this size
# LARGE dataset: 400 images

SIZE = "TINY"
PREFIX = "bdd"

if SIZE == "TINY":
  NUM_EXAMPLES = 50
  SPLITS = {"train" : 40, "test" : 10}
elif SIZE == "MEDIUM":
  NUM_EXAMPLES = 200
  SPLITS = {"train" : 160, "test" : 40}
elif SIZE == "LARGE":
  NUM_EXAMPLES = 400
  SPLITS = {"train" : 320, "test" : 80}

# set globals
IMAGE_ROOT = "data/lung_segmentation/images/training"
LABEL_ROOT = "data/lung_segmentation/annotations/training"

# set global Artifact names (allowing steps to be rerun independently)
RAW_DATA_AT = "_".join([PREFIX, "raw_data", str(NUM_EXAMPLES)])
TRAIN_DATA_AT = "_".join([PREFIX, "train_data", str(NUM_EXAMPLES)])
TEST_DATA_AT = "_".join([PREFIX, "test_data", str(NUM_EXAMPLES)])
BDD_CLASSES = ['background', 'lung']
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]
# Initialize the run
run = wandb.init(project='chestXray-demo',
                 job_type="upload",
                 config={
                     "num_examples": 704,
                     "num_train": 634,
                     "num_test": 70
                 })

# Setup a WandB Classes object. This will give additional metadata for visuals
class_set = wandb.Classes([{'name': name, 'id': id}
                           for name, id in zip(BDD_CLASSES, BDD_IDS)])

# Create an Artifact (versioned folder)
artifact = wandb.Artifact(name=RAW_DATA_AT, type="raw_data")

# Setup a WandB Table object to hold our dataset
columns = ["id", "raw", "annotated", "color_mask", "raw_label"]
# add a column for the pixel fraction of each class label
columns.extend(["%_" + c for c in BDD_CLASSES])
table = wandb.Table(
    columns=columns
)

# temporary directory to hold intermediate visualizations
TMPDIR = "tmp_labels"
if not os.path.isdir(TMPDIR):
    os.mkdir(TMPDIR)

# Fill up the table
all_images = [f for f in os.listdir(IMAGE_ROOT)]
progress_bar = ProgressBar(wandb.config.num_train)
for ndx in range(wandb.config.num_train):
    img = all_images[ndx]
    img_file = os.path.join(IMAGE_ROOT, img)
    train_id = img.split(".")[0]
    label_file = os.path.join(LABEL_ROOT, train_id + ".png")

    # First, we will build a wandb.Image to act as our raw example object
    #    classes: the classes which map to masks and/or box metadata
    #    masks: the mask metadata. In this case, we use a 2d array w
    #           here each cell corresponds to the label (this comes directly from the dataset)
    raw_img = wandb.Image(img_file)
    annotated = wandb.Image(img_file, classes=class_set,
                            masks={"ground_truth": {"mask_data": np.array(Image.open(label_file)) * 255}})

    # Next, we create an additional image which may be helpful during analysis
    # and compute the fraction of each image covered by each of the classes
    # (so we can find examples with more pixels of cars vs pedestrians vs other
    # classes of interest). This additional metadata is optional
    color_mask = util.static_label(label_file, train_id)
    class_fractions = util.count_pixels(label_file)
    raw_label = wandb.Image(label_file)

    # use .add_file for the files we need to reference by path
    # (in this case the training image and the label)
    artifact.add_file(img_file, os.path.join("images", img))
    artifact.add_file(label_file, os.path.join("labels", train_id + "_train_id.png"))

    # Finally, we add a row of our newly constructed data.
    row = [train_id, raw_img, annotated, color_mask, raw_label]
    row.extend(class_fractions)
    table.add_data(*row)
    progress_bar.update()

# .add the table to the artifact
artifact.add(table, "raw_examples")

# Finally, log the artifact
print("Saving data to WandB...")
run.log_artifact(artifact)
run.finish()
print("... Run Complete")

# clear out the temporary files
shutil.rmtree(TMPDIR)