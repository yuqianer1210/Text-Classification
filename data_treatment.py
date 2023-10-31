import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pylab, mlab
import matplotlib.pyplot as plt


train_data = pd.read_csv('dataset/ChnSentiCorp/train.tsv', sep='\t', encoding='utf-8')
print(train_data.head())
