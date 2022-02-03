from os.path import join

'''Network files for the initial ICCPS submission.'''
DOUBLE_LOOP = join(
    "configs", "ICCPS", "double_loop", "double.net.xml"
)
GRID_3x3 = join(
    "configs", "ICCPS", "grid_3x3", "grid-3x3.net.xml"
)
GRID_5x5 = join(
    "configs", "ICCPS", "grid_5x5", "grid-5x5.net.xml"
)
GRID_7x7 = join(
    "configs", "ICCPS", "grid_7x7", "grid-7x7.net.xml"
)
GRID_9x9 = join(
    "configs", "ICCPS", "grid_9x9", "grid-9x9.net.xml"
)

BOSTON_DEPR = join(
    "configs", "ICCPS", "__old", "boston_inter", "boston.net.xml"
)
COMPLEX_DEPR = join(
    "configs", "ICCPS", "__old", "complex_inter", "complex_inter.net.xml"
)
SINGLE_LOOP_DEPR = join(
    "configs", "ICCPS", "__old", "single_loop", "single.net.xml"
)


'''Network files for the first resubmission.'''
V2_GRID = join(
    "configs", "SMARTCOMP", "grid.net.xml"
)
V2_SPIDER = join(
    "configs", "SMARTCOMP", "spider.net.xml"
)
V2_RANDOM = join(
    "configs", "SMARTCOMP", "rand.net.xml"
)
