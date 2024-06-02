"""
Constant definitions for SDXL
"""

# aspect ratio (w/h), width, height
SDXL_ASPECT_RATIO_RESOLUTIONS_MAP = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

SDXL_RESOLUTIONS = [v for v in SDXL_ASPECT_RATIO_RESOLUTIONS_MAP.values()]

SDXL_SAMPLER_NAME_LIST = \
[
    "EulerEDM",
    "HeunEDM",
    "EulerAncestral",
    "DPMPP2SAncestral",
    "DPMPP2M",
    "LinearMultistep"
]

DISCRETIZATION_LIST = ["LegacyDDPMDiscretization", "EDMDiscretization"]
GUIDER_LIST = ["VanillaCFG", "IdentityGuider", "LinearPredictionGuider", "TrianglePredictionGuider"]
