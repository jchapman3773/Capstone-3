import numpy as np
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

Classes = ['person','banana']
Colors = np.random.uniform(0, 255, size=(len(Classes), 3))
