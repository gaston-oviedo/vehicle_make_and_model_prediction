import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# Assign the loaded detection model to global variable DET_MODEL
DET_MODEL = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"


def get_vehicle_coordinates(img):

    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Parameters
    ----------
    img : numpy.ndarray
       Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
       Tuple having bounding box coordinates as (left, top, right, bottom).
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(DET_MODEL))
    cfg.MODEL.DEVICE = "cpu"  # "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DET_MODEL)

    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)

    classes = outputs["instances"].pred_classes.cpu().numpy()
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    boxes_area = outputs["instances"].pred_boxes.area().cpu().numpy()

    if len(classes) == 0:
        box_coordinates = (0, 0, img.shape[1], img.shape[0])
    else:
        # removing incorrected classes
        class_list = [2, 7]  # 2:car, 7:truck
        idx_list = []
        for i in range(len(classes)):
            if classes[i] not in class_list:
                idx_list.append(i)

        boxes_filtered = np.delete(boxes, idx_list, 0)
        boxes_area_filtered = np.delete(boxes_area, idx_list, None)

        if (
            len(boxes_filtered) == 0 or len(boxes_area_filtered) == 0
        ):  # in case it detects an object different than car/truck
            box_coordinates = (0, 0, img.shape[1], img.shape[0])
        else:
            # taking the bigger element
            idx_bigger_area = np.argmax(boxes_area_filtered)

            box_coordinates = tuple(boxes_filtered[idx_bigger_area].astype(int))

    return box_coordinates
