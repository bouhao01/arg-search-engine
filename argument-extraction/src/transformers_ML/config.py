import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import transformers
from transformers import AutoModel, BertTokenizerFast, BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import time
import datetime
import random
import os
import sys
import json



# specify GPU
device = torch.device("cuda")
# device = torch.device("cpu")