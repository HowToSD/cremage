"""
Defines constants
"""
MODE_TEXT_TO_IMAGE = 1
MODE_IMAGE_TO_IMAGE = 2
MODE_INPAINTING = 3

THUMBNAIL_IMAGE_EDGE_LENGTH = 256
MAIN_IMAGE_CANVAS_SIZE = 768
VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH = 256
VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT = 144

TRUE_FALSE_LIST = ["True", "False"]

FACE_MODEL_NAME = "ip-adapter-faceid-plusv2_sd15.bin"

GMT_SD_1_5 = "SD 1.5"
GMT_SDXL = "SDXL"
GMT_SD_3 = "SD 3"
GMT_KANDINSKY_2_2 = "Kandinsky 2.2"
GMT_PIXART_SIGMA = "Pixart Sigma"
GMT_HUNYUAN_DIT = "Hunyuan-DiT"
GMT_STABLE_CASCADE = "Stable Cascade"

GENERATOR_MODEL_TYPE_LIST = [
    GMT_SD_1_5,
    GMT_SDXL,
    GMT_SD_3 ,
    GMT_STABLE_CASCADE,
    GMT_KANDINSKY_2_2,
    GMT_PIXART_SIGMA,
    GMT_HUNYUAN_DIT
]