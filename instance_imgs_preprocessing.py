# %%
import glob
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

config = OmegaConf.load("/dreambooth/config.yaml")


def load_models(
    detection_model_name: str = "IDEA-Research/grounding-dino-base",
    segmentation_model_name: str = "facebook/sam-vit-huge",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        detection_model_name, device_map=device
    )
    detection_processor = AutoProcessor.from_pretrained(detection_model_name)

    segmentation_model = SamModel.from_pretrained(
        segmentation_model_name, device_map=device
    )
    segmentation_processor = SamProcessor.from_pretrained(segmentation_model_name)
    return (
        detection_processor,
        detection_model,
        segmentation_processor,
        segmentation_model,
    )


def segmentation(image, box):
    inputs = segmentation_processor(
        image,
        input_boxes=box,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = segmentation_model(**inputs)
    masks = segmentation_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    max_idx = torch.argmax(scores)
    img_mat = np.array(image)
    mask = masks[0][0].permute([1, 2, 0]).numpy()
    img_mat[~mask[:, :, max_idx]] = 255
    return img_mat


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
        detection_processor,
        detection_model,
        segmentation_processor,
        segmentation_model,
    ) = load_models()
    os.makedirs(config["preprocessing_config"]["instance_img_dir"], exist_ok=True)
    text_labels = [[config["preprocessing_config"]["description"]]]
    img_files = glob.glob(f'{config["preprocessing_config"]["raw_instance_imgs"]}/*.*')
    logging.info(f"Total {len(img_files)} img_files.")
    for f in img_files:
        logging.info(f"Start file {f}")
        image = Image.open(f)
        inputs = detection_processor(
            images=image, text=text_labels, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = detection_model(**inputs)
        results = detection_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=config["preprocessing_config"]["grounding_dino_score_threshold"],
            text_threshold=0.5,
            target_sizes=[image.size[::-1]],
        )[0]
        logging.info(f"{len(results['boxes'])} detection boxes")
        name = f.split("/")[-1]
        for idx, (box, score, labels) in enumerate(
            zip(results["boxes"], results["scores"], results["labels"])
        ):
            box = box.view(1, 1, -1).cpu()
            img_mat = segmentation(image, box)
            Image.fromarray(img_mat).save(
                f'{config["preprocessing_config"]["instance_img_dir"]}/{idx}_{name}'
            )
# %%
