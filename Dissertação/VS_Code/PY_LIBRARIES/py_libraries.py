################################################################################################################################################
'''Python Libraries'''
################################################################################################################################################
# --data_cleaning

import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime 
from scipy.stats import zscore

#--missing_values

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer 

#--analize_outlier

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

#--outliers

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm



#--others

from IPython.display import display, HTML
