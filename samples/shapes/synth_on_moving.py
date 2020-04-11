"""
creator of synthetic data.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Updated by Itamar Gruber & Shai NG for PDL1Net project - April 2020
"""
import os
import sys
import math
import numpy.random as random
import numpy as np
import cv2
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "pdl1"

    # Train on 1 GPU and 2 images per GPU. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 classes

    # Use small images for faster training. Set the limits of the small side and
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256 * 3
    IMAGE_MAX_DIM = 256 * 3

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Set training ROIs per image due to the image size and number of objects.
    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles, ellipse) placed randomly on a cells image(IHC WSI patch) surface.
    The images are generated on the fly.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("pdl1", 1, "pos")
        self.add_class("pdl1", 2, "neg")
        self.add_class("pdl1", 3, "inf")
        self.add_class("pdl1", 4, "other")

        self.PATH_TO_IMAGE_DIR = r'D:\Nati\Itamar_n_Shai\Datasets\DataCCD\DataMaskRCNN_copy\train'
        self.PATH_SAVE_DATA = r"D:\Nati\Itamar_n_Shai\Datasets\DataSynth"
        ls_dir_images = os.listdir(self.PATH_TO_IMAGE_DIR)

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            while os.path.isdir(ls_dir_images[i]):
                i += 1
            path_to_image = os.path.join(self.PATH_TO_IMAGE_DIR, ls_dir_images[i])
            shapes = self.random_image(height, width)
            self.add_image("pdl1", image_id=i, path=path_to_image,
                           width=width, height=height,
                           shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file,
        in this case it also generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = Image.open(info['path'])
        image = image.copy()  # ?
        image = np.array(image.resize([info['height'], info['width']]))
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        image_path = os.path.join(self.PATH_SAVE_DATA, "train", '{:05d}'.format(image_id))
        image_path += ".jpeg"
        cv2.imwrite(image_path, image)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pdl1":
            return info['shapes']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 255)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])

        for i, (shape, _, dims) in enumerate(info['shapes']):
            path = os.path.join(self.PATH_SAVE_DATA, "mask_train", '{:05d}'.format(image_id))
            path += "_" + shape + "_" + '{:05d}'.format(i) + ".png"
            cv2.imwrite(path, mask[:, :, i])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'pos':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "neg":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "inf":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        elif shape == "other":
            image = cv2.ellipse(image, (x, y), (s, s * 2 // 3), 0, 0, 360, color, thickness=-1)

        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three values:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        * @param buffer is the minimum size and coordinate of the generated shape
        """
        # Shape
        shape = random.choice(["pos", "neg", "inf", "other"])
        # Color
        import colorsys
        color = colorsys.hsv_to_rgb(random.rand(), 1.0, 1.0)  # ensures no brown color is chosen
        color = tuple([int(color[i]*255) for i in range(3)])  # convert color to 255 integer dim'
        # color = tuple([random.randint(0, 255) for _ in range(3)])  # totally random color
        # Center x, y
        buffer = 50
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns a list of shape specifications that can be used to draw the image.
        """
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = 2  # random.randint(4, 6)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression with 1 threshold to avoid
        # shapes covering each other (it was 0.3)
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 1)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return shapes


if __name__ == "__main__":
    # Root directory of the project
    ROOT_DIR = os.path.abspath("../../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn.model import log

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = ShapesConfig()
    config.display()

    # Training dataset
    dataset_train = ShapesDataset()
    num_images_train = 145
    dataset_train.load_shapes(num_images_train, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset()
    num_images_val = 50
    dataset_val.load_shapes(num_images_val, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


    class InferenceConfig(ShapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    NUM_LOOP = 5
    for _ in range(NUM_LOOP):
        image_id = random.choice(dataset_val.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)

        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset_train.class_names, figsize=(8, 8))

        results = model.detect([original_image], verbose=1)

        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_val.class_names, r['scores'])  # , ax=get_ax())
