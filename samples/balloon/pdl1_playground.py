"""
Mask R-CNN
Train on the toy PDL1 dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=imagenet

    # Apply color splash to an image
    python3 pdl1_playground.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 pdl1_playground.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith(os.path.normpath("samples/balloon")):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.join(ROOT_DIR, "..", "..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class PDL1NetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "PDL1"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet50"


############################################################
#  Dataset
############################################################

class PDL1NetDataset(utils.Dataset):

    def load_pdl1net_dataset(self, dataset_dir, subset):
        """Load a subset of the PDL1 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("PDL1", 1, "inflammation")
        self.add_class("PDL1", 2, "negative")
        self.add_class("PDL1", 3, "positive")
        # if we decide to delete the next line reduce the number of classes in the config
        self.add_class("PDL1", 4, "other")

        ids = [c["id"] for c in self.class_info]
        names = [c["name"] for c in self.class_info]
        self.class_name2id = dict(zip(names, ids))

        # Train or validation dataset?
        # TODO: change the path to the right one
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # TODO: make sure the json has the right name
        # ATTENTION! the parser will work only for via POLYGON segmented regions
        # annotations = json.load(open(os.path.join(dataset_dir, "train_synth_via_json.json")))
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        # type2class = {"1":"inflammation", "2":"negative", "3":"positive", "4":"other"}
        type2class = {"inf": "inflammation", "neg": "negative", "pos": "positive", "other": "other"}
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            classes = [r['region_attributes']['category'] for r in a['regions']]  # validate that a list of classes is obtained
            classes = [type2class[c] for c in classes]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "PDL1",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a PDL1 dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "PDL1":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # TODO: make sure no intersection are made between polygons
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['all_points_y'] is None or p['all_points_x'] is None:
                continue
            #  check if an element in the list is also a list
            if any(isinstance(elem, list) for elem in p['all_points_y']) or any(isinstance(elem, list) for elem in p['all_points_x']):
                continue
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # mask_classes = [self.class_name2id[name] for name in self.class_names]
        mask_classes = [self.class_name2id[name] for name in info["classes"]]
        mask_classes = np.array(mask_classes, dtype=np.int32)

        # clean masks intersections
        # create united mask for each class
        united_masks = np.zeros([info["height"], info["width"], self.num_classes])
        for i in np.arange(self.num_classes):
            masks_of_same_class = mask[:, :, mask_classes == (i+1)]
            for single_mask_index in np.arange(masks_of_same_class.shape[2]):
                united_masks[:,:,i] = np.logical_or(united_masks[:,:,i], masks_of_same_class[:,:,single_mask_index])
        # clean each mask from intersections with united_masks
        classes_array = np.array([self.class_name2id[name] for name in self.class_names])
        for i in np.arange(mask.shape[2]):
            # stronger_classes = np.unique(mask_classes[mask_classes > mask_classes[i]])
            stronger_classes = classes_array[classes_array > mask_classes[i]]
            stronger_classes -= 1  # change from class number to index in united_masks (starts from 0)
            curr_mask = mask[:, :, i]
            for class_index in stronger_classes:
                curr_mask[np.logical_and(curr_mask, united_masks[:,:,class_index])] = 0
            mask[:, :, i] = curr_mask
        # remove other from masks
        mask = mask[:, :, mask_classes != self.class_name2id["other"]]
        mask_classes = mask_classes[mask_classes != self.class_name2id["other"]]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        for i in np.arange(mask.shape[2]):
            mask_copy = mask.copy()
            current_mask = mask_copy[:,:,i]
            current_mask[ current_mask == 1 ] = 255
            dir_to_save = r'D:\Nati\Itamar_n_Shai\Datasets\DataSynth\occlusion_result'
            image_path = os.path.join(dir_to_save, str(int(image_id)) + "_" + str(i) + '.png')
            cv2.imwrite(image_path, current_mask)

        return mask, mask_classes

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "PDL1":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


from imgaug import augmenters as iaa
import PIL
import matplotlib.pyplot as plt

def augmenter():
    seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    ], random_order=True) # apply augmenters in random order
    return seq


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PDL1NetDataset()
    dataset_train.load_pdl1net_dataset(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PDL1NetDataset()
    dataset_val.load_pdl1net_dataset(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    seq = augmenter()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=seq)

    # val_generator = mrcnn.data_generator(dataset_val, self.config, shuffle=True,
    #                                batch_size=self.config.BATCH_SIZE)

import visualize_pdl1

def test(model):
    class InferenceConfig(PDL1NetConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    # TODO: add test dataset
    # Validation dataset
    dataset_val = PDL1NetDataset()
    dataset_val.load_pdl1net_dataset(args.dataset, "val")
    dataset_val.prepare()

    print("start test")
    inference_config = InferenceConfig()
    list_pred_masks = []
    list_gt_masks = []
    matched_classes = []
    APs = []
    NUM_THRESH = 11
    threshes = np.linspace(0,1,NUM_THRESH)
    matrices = np.zeros((NUM_THRESH, dataset_val.num_classes, dataset_val.num_classes))
    confusstion_matrix = np.zeros((dataset_val.num_classes, dataset_val.num_classes))
    for image_id in [0]: # np.arange(dataset_val.num_images):
        # Load image and ground truth data
        image_name = dataset_val.image_info[image_id]['id']
        image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        gt_match, pred_match, overlaps = utils.compute_matches(gt_bboxes, gt_class_ids, gt_masks,
                                                            r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                            iou_threshold=0.5, score_threshold=0.0)

        matrices, threshes = visualize_pdl1.acumulate_confussion_matrix_multiple_thresh(matrices, threshes, 4,
                                                                                        gt_class_ids,
                                                                                        r["class_ids"], r["scores"],
                                                                                        overlaps, [])
        confusstion_matrix += visualize_pdl1.get_confusion_matrix(4, gt_class_ids, r["class_ids"], r["scores"],
                  overlaps, [], threshold=0.5)

        #  obtain all the elemnts in pred which have corresponding GT elemnt
        pred_match_exist = pred_match > -1
        #  retrieve the index of the GT element at the position of the correlated element in prediction
        sort_gt_as_pred = pred_match[pred_match_exist].astype(int)
        matched_classes.append(r["class_ids"][pred_match_exist])
        list_pred_masks.append(r['masks'][:,:,pred_match_exist])
        list_gt_masks.append(gt_masks[:,:,sort_gt_as_pred])

    visualize_pdl1.plot_auc_roc(matrices, threshes, positive_class_num=1)
    # row_sum = np.sum(confusstion_matrix, axis=1).reshape((-1,1)) + 1e-10
    # select_row_nonzero = np.tile(row_sum,(1, row_sum.shape[0])) > 0
    # confusstion_matrix = (confusstion_matrix * select_row_nonzero) / row_sum
    from sklearn.metrics import roc_curve
    a = roc_curve(np.array([0,1]), np.array([0,1]))
    num_classes = inference_config.NUM_CLASSES - 1
    IoU = visualize_pdl1.compute_batch_iou(num_classes, list_pred_masks, list_gt_masks, matched_classes)
    print(IoU)
    print("the confusion matrix is:\n {}".format(confusstion_matrix))
    # create new class list to replace the 'BG' with 'other'
    right_indices = [4, 1, 2, 3]
    copy_class_names = [dataset_val.class_names[i] for i in right_indices]
    visualize_pdl1.plot_confusion_matrix(confusstion_matrix[0:4, 0:4], copy_class_names)



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect PDL1.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/PDL1/dataset/",
                        help='Directory of the PDL1 dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train" or args.command == "test":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PDL1NetConfig()
    else:
        class InferenceConfig(PDL1NetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    print("create model")
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "test":
        test(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
