# roc_curve_with_confidence_intervals

# Python version


```python
import platform
print(platform.python_version())
```

    3.7.5


## Notes

Run you `jupyter notebook` positioned on the `stackoverflow` project folder.

The the following notebook cell will append to your path the current folder where the jupyter notebook is runnig, in order to be able to import `auc_delong_xu.py` script for this example.


```python
import os
import sys
import pandas as pd
import numpy as np
from sklearn import datasets

notebook_folder_path = !pwd
prj_path = os.path.abspath(os.path.join(notebook_folder_path[0], '', ''))
sys.path.append(prj_path)
print('Append to path: %s' % prj_path)
```

    Append to path: /Users/lsanchez/roc_curve_with_confidence_intervals


## The data

I used the iris dataset to create a binary classification task where the possitive class corresponds to the `setosa` class.

The `y_score` is simply the `sepal length` feature rescaled between `[0, 1]`.


```python
data = pd.DataFrame(
    datasets.load_iris().data,
    columns=datasets.load_iris().feature_names)
target = pd.Series([
    datasets.load_iris().target_names[x]
    for x in datasets.load_iris().target])

data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



The AUC and Delong Confidence Interval is calculated via the Yantex's implementation of Delong (see script: `auc_delong_xu.py` for further details)


```python
from auc_delong_xu import auc_ci_Delong

y_true = (target != 'setosa').astype(int)

y_score = data['sepal length (cm)']
y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

y_pred = (y_score > .5).astype(int)

auc, auc_var, ci = auc_ci_Delong(
    y_true=y_true,
    y_scores=y_score)

print('ROC AUC: %s, Conf.' % auc) 
print('Confidence Interval: %s (95%% confidence)' % str(ci))
```

    ROC AUC: 0.9586000000000001, Conf.
    Confidence Interval: [0.93020851 0.98699149] (95% confidence)



```python

```

# R Version

```R
version
```


                   _                           
    platform       x86_64-apple-darwin13.4.0   
    arch           x86_64                      
    os             darwin13.4.0                
    system         x86_64, darwin13.4.0        
    status                                     
    major          3                           
    minor          6.1                         
    year           2019                        
    month          07                          
    day            05                          
    svn rev        76782                       
    language       R                           
    version.string R version 3.6.1 (2019-07-05)
    nickname       Action of the Toes          


## The data

I used the iris dataset to create a binary classification task where the possitive class corresponds to the `setosa` class.

The `y_score` is simply the `sepal length` feature rescaled between `[0, 1]`.


```R
library(pROC)
library(datasets)

data(iris)

y_true = as.integer(iris$Species == 'setosa')
y_score = iris$Sepal.Length
y_score = (y_score - min(y_score)) / (max(y_score) - min(y_score))

y_pred = as.integer(y_score > .5)

roc = roc(y_true, y_score)
roc
```

    Type 'citation("pROC")' for a citation.
    
    Attaching package: ‘pROC’
    
    The following objects are masked from ‘package:stats’:
    
        cov, smooth, var
    
    Setting levels: control = 0, case = 1
    Setting direction: controls > cases



    
    Call:
    roc.default(response = y_true, predictor = y_score)
    
    Data: y_score in 100 controls (y_true 0) > 50 cases (y_true 1).
    Area under the curve: 0.9586



```R
print(ci(roc))
```

    95% CI: 0.9302-0.987 (DeLong)
