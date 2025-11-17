import math
import sys
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from plotly.subplots import make_subplots
from torch import Tensor
from torch.nn import init
from pathlib import Path
from PIL import Image

torch.manual_seed(1)

if 'google.colab' in sys.modules:
    from google.colab import output
    output.enable_custom_widget_manager()

!wget https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip
!unzip data_gsn.zip &> /dev/null
!rm data_gsn.zip
