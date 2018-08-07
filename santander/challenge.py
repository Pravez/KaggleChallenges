import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstats

path_train = "./train.csv"

df = pd.read_csv(path_train, encoding='utf-8')
df.describe()