from path import PathFinder
import os
import sys
PATHS = PathFinder()


sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())

from pyniche.data.yolo.API import YOLO_API


yolo = YOLO_API(PATHS["DIR_DATA"] / "ext_imu")



# rgba(255, 0, 0, 0.5)

import supervision as sv

c = sv.Color.from_rgb_tuple((0, 0, 0))

from supervision import Position, Color

plot_param = dict({
    "fill_color": Color.WHITE,
    "fill_opacity": 0.2,
    "box_thickness": 2,
    "text_scale": 1,
    "text_thickness": 2,
    "text_padding": 4,
    # "text_position": Position.BOTTOM_LEFT,
    "text": "cow",
})
yolo["test"][3:6]

yolo["test"].vis(6, **plot_param)
yolo["test"].save(PATHS["DIR_DATA"] / "ext_imu_resize" / "train", resize=(640, 640))

new_root = PATHS["DIR_DATA"] / "ext_imu_resize"
new_yolo = YOLO_API(new_root)
new_yolo["train"].vis(6, **plot_param)
new_yolo["train"][3:5]


testp = Path("abc.TXT")
testp.suffix
testp.name
# generate a non-ducpliate lisf of integers in a range
n = 100
import random
random.sample(range(n), n)
