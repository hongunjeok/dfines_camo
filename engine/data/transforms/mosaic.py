"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """
    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range, 
                                               scale=scaling_range, fill=fill_value)

    def _load_and_resize_samples(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = F.get_spatial_size(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = F.get_spatial_size(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def _create_mosaic(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        resized_images, resized_targets, max_height, max_width = self._load_and_resize_samples(image, target, dataset)

        # Generate mosaic
        mosaic_image, mosaic_target = self._create_mosaic(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        # mosaic_image.save(os.path.join('./vis_debug', str(datetime.now().timestamp()) + 'Mosaiced_affine_crop.jpg'))

        return mosaic_image, mosaic_target, dataset
