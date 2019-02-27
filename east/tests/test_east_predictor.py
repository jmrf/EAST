import cv2
import logging
import coloredlogs

from pprint import pformat
from east.model import EASTPredictor


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    coloredlogs.install(logger=logger,
                        level=logging.DEBUG)

    m = EASTPredictor()
    # m.load("checkpoints/east_icdar2015_resnet_v1_50_rbox")

    img = cv2.imread("/home/jose/code/Experimental/ROI-detection/images/cv_page.png")
    res = m.predict(img)

    logger.info(pformat(res))
