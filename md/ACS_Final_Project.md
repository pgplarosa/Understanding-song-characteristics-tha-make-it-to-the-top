# Anatomy of a Hit Song: Understanding the song characteristics that make it to the top


```python
display(Image('./img/banner.PNG'))
```


![png](output_1_0.png)


## Executive Summary

<p style="text-align:justify">Popular music is a bulti-million dollar industry today. However, the genre is derided as the worst kind of music for its repetitive, almost formulaic, and yet commercially successful style. The objective of this study is to get a glimpse of this formula and understand the audio features that land songs at the top of the charts. Using the 2010-2019 Spotify Hits Predictor Dataset from Kaggle, we conduct statistical tests and perform a logistic regression to find out what makes hits or flops.
<br><br>
Looking into the general composition of audio features between hits and flops, we identified significant differences in group means using Mann-Whitney U test for our non-normally distributed variables. We then assessed the validity of the model and its potential predictors by testing for multicollinearity and linearity with log-odds. Some variables were dropped to meet the assumptions of a logistic regression to allow for valid interpretations of the resulting summary statistics. After implementing a grid search over 10 trials and 10 splits, we then implemented an L1-regularized logistic regression using the C parameter with the highest accuracy.
<br><br>
The optimized logistic regression has an accuracy score of 78 percent. Based on the summary statistics, we derived the following insights:
<li>Danceability is the top positive predictor of hits songs.</li>
<li>Valence which represents the level of positivity also increases a song's likelihood of charting.</li>
<li>Instrumentalness appears to be the top yet negative predictor of hit songs, even exceeding danceability and valence combined.</li>
<br>
We have confirmed that a balance of certain features can contribute to a song's mass appeal and can land it at the top of the charts. Insights gleaned from this study can help artists and music producers find commercial success by producing songs that fit the standard features of those that land on the charts. Finally, record companies can also use this to predict the potential success of the songs they intend to release. </p>


```python
# Import libraries
import base64
import datetime as datetime
import time
from tqdm import tqdm
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import matplotlib.patches as mpatches
import statsmodels.api as sm

import pandas as pd
import sqlite3

import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup

from scipy.stats import shapiro, kstest, mannwhitneyu
from scipy.stats import probplot

tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import IPython

from IPython.core.display import HTML, Image
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    horizontal-align: middle;
    vertical-align: middle;
    margin:auto;
}
</style>

<script>
code_show=true;
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit"
value="Click here to toggle on/off the raw code."></form>
""")
```





<style>
.output_png {
    display: table-cell;
    text-align: center;
    horizontal-align: middle;
    vertical-align: middle;
    margin:auto;
}
</style>

<script>
code_show=true;
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit"
value="Click here to toggle on/off the raw code."></form>




## I. Introduction

<p style="text-align:justify"> Music is the art of arranging sounds in time to produce a composition through the elements of melody, harmony, rhythm, and timbre.$^1$ As one of the universal cultural aspects of all human societies, music seems magical in how it can relate and resonate with us on a visceral level. In fact, music is such a powerful catalyst that reactions range from humming along, to shedding tears and even, inspiring movement (physical and social). It is this power of music to evoke emotions and connect to people that has made it into the multi-billion dollar industry it is today.
<br><br>
Not all music is created equal however. There are music that resonate and there are those that just slip by. Popular music is a type of music with wide appeal and is typically distributed to a large audience.$^2$ However, pop music is usually criticized as the worst genre of music due to its 'endless recycling'--copy-paste of successful songs with some superficial variation. And this is not entirely wrong. In fact, Backstreet Boys, Taylor Swift, Maroon 5, and Ariana Grande all share one thing in common: Max Martin, a Swedish songwriter who is considered the backbone of the modern pop scene. It is these catchy hooks and magical melodies created by Max Martin and other Swedish producers like him that have been topping the charts for the past 50 years.$^3$
<br><br>
So does music's power just come down to a carefully calculated formula? To answer this, we conduct statistical analysis and predictive modelling on the song features of hit songs and flops. This study can be useful for artists and music producers who want commercial success by producing songs that fit the standard features of those that land on the charts. Recording companies can also predict the potential success of songs they intend to release, based on the song's audio features.</p>

## II. Methodology

The methodology used in the study consists of 3 major steps as outlined below:
#### A. Perform EDA to explore  audio features
In performing exploratory data analysis, modules including <b>pandas</b>, <b>matplotlib</b> and <b>seaborn</b> were used to manipulate data and generate visualizations. Specifically, we aim to glean insights on the means and distributions of audio features using radar charts, histograms and boxplots. We also determine how the statistics for these audio features vary between hit songs and flops.

#### B. Statistical Tests
Taking a step further from EDA, we test whether the means of hit songs and flops are significantly different across our set of features. For this, we use Shapiro-Wilk's to test for normality so we can determine the appropriate test of means to use. We then use Mann-Whitney U test for non-normal distributions to check whether the means of the two groups are significantly different.

We also perform statistical tests on our model features to check whether they are viable for use in a logistic regression. The assumptions of logistic regressions are as follows:$^{4,5}$
1. <b>Binary dependent variable.</b>
Logistic regressions require that the dependent variable be binary or categorical with few classes. In oir case, we only have two outcomes, 1 for hit and 0 for flop. 
1. <b>Independent observations.</b> This assumption requires that the observations should not come from repeated measurements or matched data. In our case, each datapoint is an individual song and is thus independent of other observations.
1. <b>Little to no multicollinearity.</b> Multicollinearity corresponds to the situation in which the data contains highly correlated independent variables. Allowing this to be prevalent could hurt our ability to interpret the coefficients and analyze the model. Like in linear regressions, this assumption will be tested via variance inflation factor [VIF] and correlation plots.
1. <b>Linearity with log-odds.</b>  For linear regression, linearity of independent variables and the response variable is assumed whereas the linearity of independent variables and the log odds of the response variable is the case for logistic regression. To test this, we use the seaborn function regplot to fit a logistic regression model and visually inspect each feature for an S-shaped curve.
1. <b>Large sample size.</b> Logistic regression typically requires a large sample size.  A general guideline is that you need a minimum of 10 cases with the least frequent outcome for each independent variable in your model. Our 6,398 observations well covers the no more than 15 features of our model.

#### C. Logistic Regression
Once we have established that the variables are fit for use, we run a Logistic Regression. This classification model is easy and efficient to implement as well as offers interpretability not only in terms of feature importance, but the direction and marginal magnitude of the impact as well. For feature selection and to avoid overfitting, we implement the logistic regression with an L1-regularization.The steps are as follows:

1. <b>Scaling.</b> Since the features are in different units, we use the StandardScaler to transform the data. This is also meant to improve interpretability and model accuracy.

1. <b>L1-regularization.</b>To optimize our regularization, we find the best possible C parameter through Grid Search using Repeated Stratified KFold on 10 splits and 10 repeats.

1. <b>Summary. </b>We generate summary statistics using the optimized regularization parameter and analyze the resulting coefficients.

## III. Data Processing

### A. Load Dataset

The dataset is downloaded from Kaggle's Spotify Hits Predictor Database. We selected the 2010-2019 dataset to get the most recent and relevant results. The dataset has 6,398 songs with an equal number of hit songs and flops. The data is clean and complete with no missing values.


```python
df = pd.read_csv('dataset-of-10s.csv')
display(HTML(f'''<h3 style="text-align:center"> Table 1:
                Sample Dataset
                </h3>'''))
df.head(3)
```


<h3 style="text-align:center"> Table 1:
                Sample Dataset
                </h3>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>uri</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>chorus_hit</th>
      <th>sections</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wild Things</td>
      <td>Alessia Cara</td>
      <td>spotify:track:2ZyuwVvV6Z3XJaXIFbspeE</td>
      <td>0.741</td>
      <td>0.626</td>
      <td>1</td>
      <td>-4.826</td>
      <td>0</td>
      <td>0.0886</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.0828</td>
      <td>0.706</td>
      <td>108.029</td>
      <td>188493</td>
      <td>4</td>
      <td>41.18681</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Surfboard</td>
      <td>Esquivel!</td>
      <td>spotify:track:61APOtq25SCMuK0V5w2Kgp</td>
      <td>0.447</td>
      <td>0.247</td>
      <td>5</td>
      <td>-14.661</td>
      <td>0</td>
      <td>0.0346</td>
      <td>0.871</td>
      <td>0.814</td>
      <td>0.0946</td>
      <td>0.250</td>
      <td>155.489</td>
      <td>176880</td>
      <td>3</td>
      <td>33.18083</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Love Someone</td>
      <td>Lukas Graham</td>
      <td>spotify:track:2JqnpexlO9dmvjUMCaLCLJ</td>
      <td>0.550</td>
      <td>0.415</td>
      <td>9</td>
      <td>-6.557</td>
      <td>0</td>
      <td>0.0520</td>
      <td>0.161</td>
      <td>0.000</td>
      <td>0.1080</td>
      <td>0.274</td>
      <td>172.065</td>
      <td>205463</td>
      <td>4</td>
      <td>44.89147</td>
      <td>9</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(HTML(f'''<h3 style="text-align:center"> Table 2:
                Dataset has no null values
                </h3>'''))
pd.DataFrame(df.isnull().sum()).T
```


<h3 style="text-align:center"> Table 2:
                Dataset has no null values
                </h3>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>uri</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>chorus_hit</th>
      <th>sections</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### B. Data Descriptions

<p style="text-align:justify"> The dataset consists of audio features that describe each song’s sonic quality, from relatively objective musical characteristics, such as <code>key</code>, <code>mode</code>, and <code>tempo</code>, to more perceptual features that quantify a song’s <code>acousticness</code>, <code>energy</code>, and <code>danceability</code>, among others. The target is a binary variable where 1' implies that the song has been featured in the weekly list of the Billboard Hot-100 tracks in that decade at least once and is therefore a 'hit'. Meanwhile, '0' implies that the track is a 'flop' given the following conditions:$^6$</p>

- The track must not appear in the 'hit' list of that decade.

- The track's artist must not appear in the 'hit' list of that decade.

- The track must belong to a genre that could be considered non-mainstream and / or avant-garde. 

- The track's genre must not have a song in the 'hit' list.

- The track must have 'US' as one of its markets.

The variables are further described in the table below.$^7$

#### <center>Table 3: Description of Variables</center>
| Column | Data type | Description | 
| :---- | :---------: | :----- |
| `year` | integer | Year when the song was in the top 100 in the billboard site |
| `song_rank` | integer | Rank of the song during the year it was in the top 100 |
| `title` | string | Title of the song |
| `artist` | string | Artist of the song in Billboard |
| `spotify_id` | string | Song ID in Spotify |
| `spotify_track` | string | Track where the song is placed in Spotify |
| `spotify_artist` | string | Name of the artist in Spotify |
| `danceability` | float | Measure of how suitable a song is to dance to |
| `energy` | float | The song's "intensity" |
| `key` | int | Estimated overall key of the track. Integers map to pitches using standard pitch class notation |
| `loudness` | float | Magnitude of auditory sensation produced depending on the amplitude of the sound wave |
| `mode` | bool | Indicates the modality (major=1 or minor=0) of a track, the type of scale of its melodic content |
| `speechiness` | float | Measure/amount of spoken words vs singing in in the track |
| `acousticness` | float | How likely the algorithm thinks the song is recorded without electronic instruments or effects |
| `instrumentalness` | float | Estimated likelihood that the song is instrumental |
| `liveliness` | float | How likely the algorithm thinks the song is recorded live |
| `valence` | float | Represents "positiveness" / vibe or how happy it makes the listener  |
| `tempo` | float | Speed of the song |
| `duration_ms` | float | Duration of the song in milliseconds |
| `time_signature` | int | Specify how many beats (pulses) are to be contained in each bar and which note value is to be given one beat |
|`chorus_hit`|float|timestamp of the start of the chorus of the track (in milliseconds)|
|`sections`|int|number of sections|

Note: Feature values are based on Spotify's own calculations.

### D. Data Exploration

<p style="text-align:justify">We begin the analysis by exploring song features between hit songs and flop songs.</p>


```python
#standardize
radar = df.iloc[:,3:]
st = (radar-radar.min(axis=0))/(radar.max(axis=0)-radar.min(axis=0))

#separate 
mean_st0 = st[st.target==0].iloc[:,:-1].mean().reset_index()
mean_st1 = st[st.target==1].iloc[:,:-1].mean().reset_index()

#for graph
mean_st0 = mean_st0.append(mean_st0.iloc[0,:])
mean_st1 = mean_st1.append(mean_st1.iloc[0,:])
```


```python
# between two groups
lbl = list(mean_st0['index'].values)
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(mean_st0))
plt.figure(figsize=(6,6))
ax = plt.subplot(polar=True)

#graph title
display(HTML(f'''<h3 style="text-align:center"> Figure 1:
                Hit vs Flop Averages
                </h3>'''))

#overall
ax.plot(label_loc, mean_st1[0], '#008B45', label='Hit', linewidth=0.2)
ax.fill(label_loc, mean_st1[0], '#008B45', alpha=.5)
ax.plot(label_loc, mean_st0[0], 'gray', label='Flop', linewidth=0.2)
ax.fill(label_loc, mean_st0[0], 'gray', alpha=.5)

ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
lines, labels = plt.thetagrids(np.degrees(label_loc),
                               labels=lbl)
plt.tick_params(axis='both', which='major', pad=10)
plt.legend(bbox_to_anchor=(1.15, 1.05), loc='upper right')
plt.ylim((0,1.1))
plt.box(on=None)
```


<h3 style="text-align:center"> Figure 1:
                Hit vs Flop Averages
                </h3>



![png](output_20_1.png)


Looking at group averages (Figure 1), hit songs appear distinct from flops as they have higher valence, danceability and loudness. Hit songs also have lower instrumentalness and acousticness compared to flops.


```python
#graph title
display(HTML(f'''<h3 style="text-align:center"> Figure 2:
                Hit vs Flop Distribution
                </h3>'''))

# kde between 2 groups
plt.rcParams["figure.figsize"] = (10, 20)
test = df.iloc[:,3:]
for i,k in enumerate(radar.columns[:-1]):
    legend = True if i==1 else False
    ax = plt.subplot(8,2,i+1)
    ax = plt.hist(radar[radar.target == 1][k], alpha=0.5, color='#008B45')
    ax = plt.hist(radar[radar.target == 0][k], alpha=0.5, color='gray')
    if legend==True:
        plt.legend(bbox_to_anchor=(1.5, 1.05), labels=['Spacing','Limiting','Met'])
    plt.title(k)
    plt.tight_layout()
```


<h3 style="text-align:center"> Figure 2:
                Hit vs Flop Distribution
                </h3>



![png](output_22_1.png)


Figure 2 shows the distribution of each song feature for the hit and flop groups. Most distributions appear skewed with the exception of danceability and tempo. The flop group also appears to have more extreme values which will be explored further in the next charts.


```python
#graph title
display(HTML(f'''<h3 style="text-align:center"> Figure 3:
                Hit vs Flop Distribution
                </h3>'''))
plt.rcParams["figure.figsize"] = (10, 20)
for i,k in enumerate(test.columns[:-1]):
    ax = plt.subplot(8,2,i+1)
    sns.boxplot(data=test,
                x='target',
                y=k,
                ax=ax, palette='Greens')
    ax.set_xlabel(None)
    ax.set_xticklabels(['Flop','Hit'])
plt.show()
```


<h3 style="text-align:center"> Figure 3:
                Hit vs Flop Distribution
                </h3>



![png](output_24_1.png)


As we have seen in the distribution plots, significant outliers (Figure 3) appear for many of the features including loudness, speechiness, liveness, duration, time signature, sections, and chorus hit. Meanwhile, only the hit group has outliers for acousticness and instrumentalness.

## IV. Results and Discussion

<p style="text-align:justify">In this section, we will conduct statistical tests on the audio features as well as test the validity of these features for use in a L1-regularized logistic regression. </p>

### A. Tests for normality

<p style="text-align:justify">Test of normality is an important step for deciding the statistical procedure needed to compare groups. It will become the basis of decision whether to use a parametric test for normally distributed feature or a non-parametric test otherwise. The two most popular statistical tests of normality are <i>Shapiro Wilk</i> and <i>Kolmogorov-Smirnov </i>.</p>

<ul>
    <li><i>Shapiro Wilk</i> - based on the correlation between the sample percentile and normal percentile</li>
    <li><i>Kolmogorov-Smirnov</i> - based on the correlation between the sample cumulative distribution function and some theoretical cumulative distribution function<sup>11</sup></li>
</ul>

<p style="text-align:justify">For some literatures, it is always best to try <i>Shapiro Wilk</i> as it has more power than that of <i>Kolmogorov-Smirnov</i>. However, when the sample size is greater than 50, it is also good to check on the <i>Kolmogorov-Smirnov</i><sup>12</sup>. One problem for both is that they are prone to commit Type II error for very large sample sizes<sup>13</sup>. In this study, we explored using both of these statistical tests and verified the results by using q-q plots.</p>

<p style="text-align:justify">Formally, we represent these hypotheses as: </p>
<center><b><i>H<sub>0</sub></i></b> : data is normally distributed</center>
<center><b><i>H<sub>1</sub></i></b> : data is not normal distributed</center>


```python
def normality(df, alpha=0.05):
    out = pd.DataFrame()
    
    for col in df.columns[:-1]:
        for i in range(2):
            # categorical/ordinal variables
            if col in ['key', 'mode', 'time_signature']:
                continue
            
            stat_sh, pvalue_sh = shapiro(df.loc[df.target==i, col].to_numpy())
            stat_ks, pvalue_ks = kstest(df.loc[df.target==i, col].to_numpy(),
                                       'norm')
            
            if pvalue_sh < alpha:
                decision_sh = "Reject H0"
            else:
                decision_sh = "Do Not Reject H0"
            
            if pvalue_ks < alpha:
                decision_ks = "Reject H0"
            else:
                decision_ks = "Do Not Reject H0"
            
            columns = pd.MultiIndex.from_product([['Shapiro Wilk', 
                                                   'Kolmogorov-Smirnov'], 
                                                  ['statistics', 'p-value', 
                                                   'alpha', 'decision']])
            
            data = np.array([round(stat_sh, 3), 
                             '{:.2e}'.format(float(pvalue_sh)), 
                             alpha, decision_sh,
                             round(stat_ks, 3), 
                             '{:.2e}'.format(float(pvalue_ks)), 
                             alpha, decision_ks])[None, :]
            
            
            index = pd.MultiIndex.from_product([[col], [i]],
                                   names=['feature', 'target'])
            
            temp = pd.DataFrame(data=data, columns=columns, index=index)

            out = pd.concat([out, temp], axis=0)
    
    return out

display(HTML(f'''<h3 style="text-align:left"> Table 4:
                Tests for Normality
                </h3>'''))
display(normality(df.loc[:, 'danceability':]))
```


<h3 style="text-align:left"> Table 4:
                Tests for Normality
                </h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">Shapiro Wilk</th>
      <th colspan="4" halign="left">Kolmogorov-Smirnov</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>statistics</th>
      <th>p-value</th>
      <th>alpha</th>
      <th>decision</th>
      <th>statistics</th>
      <th>p-value</th>
      <th>alpha</th>
      <th>decision</th>
    </tr>
    <tr>
      <th>feature</th>
      <th>target</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">danceability</th>
      <th>0</th>
      <td>0.983</td>
      <td>4.51e-19</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.531</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.996</td>
      <td>1.53e-07</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.613</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">energy</th>
      <th>0</th>
      <td>0.897</td>
      <td>5.64e-42</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.5</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.97</td>
      <td>1.67e-25</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.6</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">loudness</th>
      <th>0</th>
      <td>0.807</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.97</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.926</td>
      <td>4.72e-37</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.976</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">speechiness</th>
      <th>0</th>
      <td>0.657</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.509</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.736</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.509</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">acousticness</th>
      <th>0</th>
      <td>0.738</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.5</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.756</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.5</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">instrumentalness</th>
      <th>0</th>
      <td>0.74</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.5</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.095</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.5</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">liveness</th>
      <th>0</th>
      <td>0.734</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.51</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.766</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.508</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">valence</th>
      <th>0</th>
      <td>0.946</td>
      <td>1.25e-32</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.509</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.984</td>
      <td>1.60e-18</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.522</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">tempo</th>
      <th>0</th>
      <td>0.982</td>
      <td>1.24e-19</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>1.0</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.984</td>
      <td>8.17e-19</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>1.0</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">duration_ms</th>
      <th>0</th>
      <td>0.831</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>1.0</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.918</td>
      <td>1.26e-38</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>1.0</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">chorus_hit</th>
      <th>0</th>
      <td>0.858</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.998</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.844</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>1.0</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sections</th>
      <th>0</th>
      <td>0.815</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.996</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.968</td>
      <td>3.05e-26</td>
      <td>0.05</td>
      <td>Reject H0</td>
      <td>0.999</td>
      <td>0.00e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">The results for both <i>Shapiro Wilk</i> and <i>Kolmogorov-Smirnov</i> (Table 4) with 95% confidence interval show that all of our continuous features do not follow a normal distribution. Visually, we can check the q-q plots to assess if our statistical tests are just being sensitive. q-q plot is a visual representation that lets us know the if the data follows a certain distribution by means of quantiles<sup>14</sup>.</p>


```python
def plot_qq(df):
    fig, axes = plt.subplots(len(df.columns) -4, 2,
                             figsize=(20, 40), sharex=True)
    axes = axes.flatten()
    
    cnt = 0
    for col in df.columns[:-1]:
        # categorical/ordinal variables
        if col in ['key', 'mode', 'time_signature']:
                continue
        
        for i in range(2):
            label = ['Flop','Hit']
            ax = axes[cnt]
            test_df = df.loc[df.target==i, col].to_numpy()
            probplot(test_df, dist = "norm", plot = ax)
            ax.get_children()[0].set_color('#008B45')
            ax.get_children()[1].set_color('gray')
            ax.set_title(f'{col}: {label[i]}')
            cnt += 1
    plt.tight_layout()
    return ax

display(HTML(f'''<h3 style="text-align:center"> Figure 4:
                Q-Q plots
                </h3>'''))
plot_qq(df.loc[:, 'danceability':]);
```


<h3 style="text-align:center"> Figure 4:
                Q-Q plots
                </h3>



![png](output_32_1.png)


<p style="text-align:justify">We could say that our data follows a normal distribution if the sample quantiles (scatter) are fitted in the normal quantiles (line). Looking at Figure 4, we can see that all of our features separate at the tail end. Danceability and tempo seem to be closest to having normal distribution but showed some level of thinness of tails<sup>15</sup>. With this information, we could say that our statistical tests are indeed correct that none our continuous features follow a normal distribution. Therfore, a <b>non-parametric</b> test is needed to compare the two groups.</p>

### B. Mann-Whitney U Test

<p style="text-align:justify">In this study, we used Mann-Whitney U test to compare two sample means. It is a non-parametric test which means it does not assume any assumption with regards to distribution. However, the test does require some assumptions to be made.</p>

<b>Assumptions of Mann Whitney U<sup>16</sup>:</b>
<ol>
    <li>dependent variables should be measured at the ordinal or continuous level</li>
    <li>independent variables should consist of two categorical, independent groups.</li>
    <li>independence of observations</li>
    <li>not normally distributed</li>
 </ol>
 
<p style="text-align:justify">All of these assumptions are met with our data: our dependent variables are all ordinal or continuous (danceability, tempo, etc); our independent variable consists of a categorical group (hit or not hit); each observation is an individual song thus independent of others; and finally, our data is not normally distributed hence our use of non-parametric test.</p>
 
 <p style="text-align:justify">Formally, we represent these hypotheses as: </p>
<center><b><i>H<sub>0</sub></i></b> : the distributions of "hit" and "flop" are equal</center>
<center><b><i>H<sub>1</sub></i></b> : the distributions of "hit" and "flop" are not equal</center>


```python
def mean_test(df, alpha=0.05):
    out = pd.DataFrame()
    
    for col in df.columns[:-1]:
        negative_class = df.loc[df.target==0, col].to_numpy()
        positive_class = df.loc[df.target==1, col].to_numpy()
        
        stat, pvalue = mannwhitneyu(negative_class, 
                                    positive_class, 
                                    alternative='two-sided')
        if pvalue < alpha:
            decision = "Reject H0"
        else:
            decision = "Do Not Reject H0"
        temp = pd.DataFrame({'column' : col,
                             'statistics': stat,
                             'p-value' : pvalue,
                             'alpha' : alpha, 
                             'decision': decision}, index=[0])

        out = pd.concat([out, temp], axis=0)
    
    return out.reset_index(drop=True)


display(HTML(f'''<h3 style="text-align:left"> Table 5:
                Mann-Whitney U Test Results
                </h3>'''))
display(mean_test(df.loc[:, 'danceability':]))
```


<h3 style="text-align:left"> Table 5:
                Mann-Whitney U Test Results
                </h3>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>statistics</th>
      <th>p-value</th>
      <th>alpha</th>
      <th>decision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>danceability</td>
      <td>2964852.5</td>
      <td>1.461886e-186</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>energy</td>
      <td>5406589.0</td>
      <td>8.750881e-05</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key</td>
      <td>5083369.0</td>
      <td>6.494958e-01</td>
      <td>0.05</td>
      <td>Do Not Reject H0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loudness</td>
      <td>3464168.5</td>
      <td>7.447737e-111</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mode</td>
      <td>4960049.5</td>
      <td>1.043594e-02</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>speechiness</td>
      <td>4888182.0</td>
      <td>1.969516e-03</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>acousticness</td>
      <td>4789477.5</td>
      <td>9.382119e-06</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>instrumentalness</td>
      <td>8641926.5</td>
      <td>0.000000e+00</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>liveness</td>
      <td>5188598.5</td>
      <td>3.310854e-01</td>
      <td>0.05</td>
      <td>Do Not Reject H0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>valence</td>
      <td>3851646.0</td>
      <td>9.443943e-66</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tempo</td>
      <td>4854673.0</td>
      <td>3.875848e-04</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>duration_ms</td>
      <td>5921695.0</td>
      <td>1.207043e-27</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>time_signature</td>
      <td>4709188.0</td>
      <td>8.715692e-28</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>chorus_hit</td>
      <td>5518585.0</td>
      <td>5.360339e-08</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sections</td>
      <td>5477793.0</td>
      <td>8.828848e-07</td>
      <td>0.05</td>
      <td>Reject H0</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">From the results of our Mann-Whitney U Test (Table 5), with 95% confidence level, we can infer the following insights:</p>
<ul>
    <li>key and liveness have equal distributions</li>
</ul>


### C. Multicollinearity
Multicollinearity between variables can cause logistic regression to have unstable estimates and inaccurate variances thus, we would like this to be as low as possible.

##### Correlation heatmap

A quick look at the correlation heatmap (Figure 5) reveals some highly-correlated variables. For instance, duration_ms and sections show a strong positive correlation coefficient of 0.81. On the other hand, significant negative correlation can be seen between energy and acousticness and also between loudness and acousticness. These strong correlations suggest that a deeper multicollinearity assessment should be performed on the data to make it suitable for logistic regression modeling.


```python
#correlation plot
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    name='test', 
    colors=['red','white','green']
)

X_mc = df.loc[:, 'danceability': 'sections']
corr = X_mc.corr()
fig, ax = plt.subplots(1, 1, figsize=(10,10))
cbar_ax = fig.add_axes([.92, .15, .05, .7])
display(HTML(f'''<h3 style="text-align:center">Figure 5. Correlation plots
                    between variables</h3>'''))
sns.heatmap(corr, ax=ax,
            vmin=-1, vmax=1, center=0,
           cmap=cmap, square=True, annot=True, fmt=".2f",#linewidths=.5,
           # xticklabels=cols,
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "vertical"});
```


<h3 style="text-align:center">Figure 5. Correlation plots
                    between variables</h3>



![png](output_41_1.png)


##### Variance Inflation Factor

For testing multicollinearity, we will be using Variance Inflation Factor (VIF) which measures the amount of multicollinearity in a set of multiple regression variables. For this test, we will adopt a VIF threshold of < 10 in accordance with generally accepted heuristics.<sup>17</sup>


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X_mc.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_mc.values, i)
                   for i in range(len(X_mc.columns))]

# Plotting VIF
display(HTML(f'''<h3 style="text-align:center">Figure 6. VIF Chart prior feature removal</h3>'''))
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(vif_data.feature[::-1].str.title(), vif_data.VIF[::-1],
         color=['green', 'red', 'green', 'green', 'green', 'green', 'green',
                'green', 'green', 'green', 'red', 'red', 'red', 'green',
                'red'][::-1])
plt.axvline(x=10, color='violet', linestyle='--')
for index, value in enumerate(vif_data.VIF[::-1]):
    plt.text(value, index, str(round(value, 2)))
plt.show()
```


<h3 style="text-align:center">Figure 6. VIF Chart prior feature removal</h3>



![png](output_44_1.png)


Looking at Figure 6, features showing high multicollinearity include most time-related features like `tempo`, `duration_ms`, `time_signature`, and `sections`. Hence, these features will be discarded to minimize multicollinearity for our data.


```python
X_mc = df.loc[:, ['danceability', 'key', 'loudness', 'mode',
                  'speechiness', 'acousticness', 'instrumentalness',
                  'liveness', 'valence']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X_mc.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_mc.values, i)
                   for i in range(len(X_mc.columns))]


# Plotting VIF
display(HTML(f'''<h3 style="text-align:center">Figure 7. VIF Chart after feature removal</h3>'''))
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(vif_data.feature[::-1], vif_data.VIF[::-1], color='green')
plt.axvline(x=10, color='violet', linestyle='--')
for index, value in enumerate(vif_data.VIF[::-1]):
    plt.text(value, index, str(round(value, 2)))
plt.show()
```


<h3 style="text-align:center">Figure 7. VIF Chart after feature removal</h3>



![png](output_46_1.png)


Removal of the identified highly multicollinear features enabled the minimization of all VIF scores to under 10, with danceability having the highest at 9.48 (Figure 7). Therefore, retention only of these remaining features would enable the data to meet the non-multicollinearity criteria of logistic regression.

### D. Test of Linearity with  Log-odds

Although logistic regression does not necessitate that continuous predictor variables be linearly related to the target variable, it does require these predictor variables to be linearly related to the log odds of the target variable. Hence, we perform a test of linearity with log-odds on the remaining continuous predictor variables after having conducted the preceding statistical tests. We do this by using the Seaborn statistical plotting library.$^{18}$


```python
df_bt=df[['danceability', 'key', 'loudness', 'mode', 'speechiness', 
          'acousticness', 'instrumentalness', 'liveness', 'valence', 
          'target']]
```


```python
test_dance = sns.regplot(x='danceability', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 8. Danceability Log Odds Linear Plot</h3>'''))
test_dance.figure.savefig("danceability log lin.png")
```


<h3 style="text-align:left">Table 8. Danceability Log Odds Linear Plot</h3>



![png](output_51_1.png)



```python
test_loudness = sns.regplot(x='loudness', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 9. Loudness Log Odds Linear Plot</h3>'''))
test_loudness.figure.savefig("loudness log lin.png")
```


<h3 style="text-align:left">Table 9. Loudness Log Odds Linear Plot</h3>



![png](output_52_1.png)



```python
test_speechiness = sns.regplot(x='speechiness', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 10. Speechiness Log Odds Linear Plot</h3>'''))
test_speechiness.figure.savefig("speechiness log lin.png")
```


<h3 style="text-align:left">Table 10. Speechiness Log Odds Linear Plot</h3>



![png](output_53_1.png)



```python
test_acousticness = sns.regplot(x='acousticness', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 11. Acousticness Log Odds Linear Plot</h3>'''))
test_acousticness.figure.savefig("acousticness log lin.png")
```


<h3 style="text-align:left">Table 11. Acousticness Log Odds Linear Plot</h3>



![png](output_54_1.png)



```python
test_instrumentalness = sns.regplot(x='instrumentalness', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 12. Instrumentalness Log Odds Linear Plot</h3>'''))
test_instrumentalness.figure.savefig("instrumentalness log lin.png")
```


<h3 style="text-align:left">Table 12. Instrumentalness Log Odds Linear Plot</h3>



![png](output_55_1.png)



```python
test_liveness = sns.regplot(x='liveness', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 13. Liveness Log Odds Linear Plot</h3>'''))
test_liveness.figure.savefig("liveness log lin.png")
```


<h3 style="text-align:left">Table 13. Liveness Log Odds Linear Plot</h3>



![png](output_56_1.png)



```python
test_valence = sns.regplot(x='valence', y='target', data=df_bt, logistic=True, color='green')
display(HTML(f'''<h3 style="text-align:left">Table 14. Valence Log Odds Linear Plot</h3>'''))
test_valence.figure.savefig("valence log lin.png")
```


<h3 style="text-align:left">Table 14. Valence Log Odds Linear Plot</h3>



![png](output_57_1.png)


To satisfy the assumption that the continuous predictor variables are linearly related to the log odds, the plots should show an S-shaped or reverse S-shaped curve - flat top and bottom with an increase/decrease in the middle. All the plots above satisfy this requirement. We can therefore say that all the remaining features fulfill the requirements for linearity with log odds.

### E. Logistic Regression

Logistic regression is a commonly-used classification algorithm when the target variable is binary (i.e. either 0 or 1). From the results of the multicollinearity tests, we have selected the following feature variables to use for our classification:

- danceability
- key
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- liveness
- valence

Upon checking these features differ in units and magnitude (Table 6), so we transform the data using the StandardScaler to improve the model's accuracy and interpretability.


```python
mm = pd.DataFrame(index=X_mc.columns, columns=['min', 'max'])
for i in X_mc.columns:
    mm.loc[i, 'min'] = min(X_mc[i])
    mm.loc[i, 'max'] = max(X_mc[i])
display(HTML(f'''<h3 style="text-align:left">Table 6. Minimum and Maximum of Predictors</h3>'''))
mm
```


<h3 style="text-align:left">Table 6. Minimum and Maximum of Predictors</h3>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>danceability</th>
      <td>0.0622</td>
      <td>0.981</td>
    </tr>
    <tr>
      <th>key</th>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>-46.655</td>
      <td>-0.149</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>0.0225</td>
      <td>0.956</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>0.0</td>
      <td>0.996</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>0.0</td>
      <td>0.995</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>0.0167</td>
      <td>0.982</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>0.0</td>
      <td>0.976</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler().fit(X_mc)
X_log = Scaler.transform(X_mc)
```

We applied Grid Search on Logistic Regression to find the best possible C parameter using Repeated Stratified KFold on 10 splits and 10 repeats. We chose to implement Logistic Regression with L1-regularization since this method leads to feature selection. Results of Grid Search over parameters

<code>C = [1e-5, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000]</code>

are given below:


```python
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

y = df['target']

C_range = [1e-5, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000]

rskf = RepeatedStratifiedKFold(n_splits=10,
                               n_repeats=10)
estimator = LogisticRegression(penalty='l1', solver='liblinear')
param_grid = {'C': C_range}
gs_lr = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=rskf)
gs_lr.fit(X_log, y);
```


```python
results_df = pd.DataFrame(gs_lr.cv_results_)
display(HTML(f'''<h3 style="text-align:left">Table 7. Results of Grid Search CV</h3>'''))
results_df
```


<h3 style="text-align:left">Table 7. Results of Grid Search CV</h3>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_C</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>...</th>
      <th>split93_test_score</th>
      <th>split94_test_score</th>
      <th>split95_test_score</th>
      <th>split96_test_score</th>
      <th>split97_test_score</th>
      <th>split98_test_score</th>
      <th>split99_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.004630</td>
      <td>0.000763</td>
      <td>0.000497</td>
      <td>0.000099</td>
      <td>0.00001</td>
      <td>{'C': 1e-05}</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>...</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.499218</td>
      <td>0.500782</td>
      <td>0.500000</td>
      <td>0.000350</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.005503</td>
      <td>0.000584</td>
      <td>0.000500</td>
      <td>0.000081</td>
      <td>0.001</td>
      <td>{'C': 0.001}</td>
      <td>0.723437</td>
      <td>0.685937</td>
      <td>0.701562</td>
      <td>0.712500</td>
      <td>...</td>
      <td>0.718750</td>
      <td>0.704688</td>
      <td>0.695312</td>
      <td>0.698438</td>
      <td>0.723437</td>
      <td>0.716745</td>
      <td>0.708920</td>
      <td>0.710847</td>
      <td>0.011409</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.007702</td>
      <td>0.000954</td>
      <td>0.000462</td>
      <td>0.000079</td>
      <td>0.01</td>
      <td>{'C': 0.01}</td>
      <td>0.781250</td>
      <td>0.748437</td>
      <td>0.750000</td>
      <td>0.778125</td>
      <td>...</td>
      <td>0.776563</td>
      <td>0.757812</td>
      <td>0.765625</td>
      <td>0.764062</td>
      <td>0.771875</td>
      <td>0.762128</td>
      <td>0.755869</td>
      <td>0.764911</td>
      <td>0.012429</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016122</td>
      <td>0.001498</td>
      <td>0.000504</td>
      <td>0.000086</td>
      <td>0.1</td>
      <td>{'C': 0.1}</td>
      <td>0.796875</td>
      <td>0.759375</td>
      <td>0.762500</td>
      <td>0.796875</td>
      <td>...</td>
      <td>0.779687</td>
      <td>0.775000</td>
      <td>0.771875</td>
      <td>0.781250</td>
      <td>0.793750</td>
      <td>0.766823</td>
      <td>0.760563</td>
      <td>0.774492</td>
      <td>0.014439</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.020239</td>
      <td>0.002483</td>
      <td>0.000489</td>
      <td>0.000075</td>
      <td>1</td>
      <td>{'C': 1}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.764062</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.779687</td>
      <td>0.775000</td>
      <td>0.773438</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.763693</td>
      <td>0.762128</td>
      <td>0.776258</td>
      <td>0.014293</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.021582</td>
      <td>0.002357</td>
      <td>0.000524</td>
      <td>0.000087</td>
      <td>10</td>
      <td>{'C': 10}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.762500</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.778125</td>
      <td>0.775000</td>
      <td>0.773438</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.765258</td>
      <td>0.762128</td>
      <td>0.776196</td>
      <td>0.014218</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.022239</td>
      <td>0.002180</td>
      <td>0.000528</td>
      <td>0.000081</td>
      <td>100</td>
      <td>{'C': 100}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.762500</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.778125</td>
      <td>0.776563</td>
      <td>0.775000</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.765258</td>
      <td>0.762128</td>
      <td>0.776258</td>
      <td>0.014223</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.022139</td>
      <td>0.002253</td>
      <td>0.000525</td>
      <td>0.000073</td>
      <td>1000</td>
      <td>{'C': 1000}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.762500</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.779687</td>
      <td>0.776563</td>
      <td>0.775000</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.765258</td>
      <td>0.762128</td>
      <td>0.776274</td>
      <td>0.014244</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.021485</td>
      <td>0.002459</td>
      <td>0.000502</td>
      <td>0.000082</td>
      <td>10000</td>
      <td>{'C': 10000}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.762500</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.779687</td>
      <td>0.776563</td>
      <td>0.775000</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.765258</td>
      <td>0.762128</td>
      <td>0.776258</td>
      <td>0.014239</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.022095</td>
      <td>0.002540</td>
      <td>0.000517</td>
      <td>0.000084</td>
      <td>100000</td>
      <td>{'C': 100000}</td>
      <td>0.793750</td>
      <td>0.757812</td>
      <td>0.762500</td>
      <td>0.803125</td>
      <td>...</td>
      <td>0.778125</td>
      <td>0.776563</td>
      <td>0.775000</td>
      <td>0.784375</td>
      <td>0.798438</td>
      <td>0.765258</td>
      <td>0.762128</td>
      <td>0.776242</td>
      <td>0.014243</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 109 columns</p>
</div>




```python
best_log = LogisticRegression(C=1_000, penalty='l1', solver='liblinear')
best_log.fit(X_log,y)
coefs_count = len(X_mc.columns)

display(HTML(f'''<h3 style="text-align:center">Figure 15. Average Coefficients</h3>'''))

fig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.barh(np.arange(coefs_count), sorted(best_log.coef_[0]), color='#008B45')
ax.set_yticks(np.arange(coefs_count))
ax.set_yticklabels(X_mc.columns[np.argsort(best_log.coef_[0])])
plt.show()
```


<h3 style="text-align:center">Figure 15. Average Coefficients</h3>



![png](output_66_1.png)



```python
X_stats = sm.add_constant(X_log)
display(HTML(f'''<h3 style="text-align:left">Table 8. Results of Logistic Regression</h3>'''))
# Use statsmodels to fit logit
logit = sm.Logit(y, X_stats).fit_regularized(method='l1', alpha=1/1_000)

# Print summary and marginal effects

print(logit.summary())
print(logit.get_margeff().summary())
```


<h3 style="text-align:left">Table 8. Results of Logistic Regression</h3>


    Optimization terminated successfully    (Exit mode 0)
                Current function value: 0.4666756210151764
                Iterations: 40
                Function evaluations: 40
                Gradient evaluations: 40
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                 target   No. Observations:                 6398
    Model:                          Logit   Df Residuals:                     6388
    Method:                           MLE   Df Model:                            9
    Date:                Wed, 15 Sep 2021   Pseudo R-squ.:                  0.3267
    Time:                        12:57:13   Log-Likelihood:                -2985.8
    converged:                       True   LL-Null:                       -4434.8
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -1.1495      0.101    -11.378      0.000      -1.348      -0.952
    x1             0.8854      0.044     20.010      0.000       0.799       0.972
    x2             0.0131      0.032      0.413      0.680      -0.049       0.075
    x3             0.8212      0.074     11.114      0.000       0.676       0.966
    x4             0.0951      0.032      2.949      0.003       0.032       0.158
    x5            -0.0310      0.031     -1.015      0.310      -0.091       0.029
    x6             0.0346      0.045      0.769      0.442      -0.054       0.123
    x7            -2.9007      0.199    -14.540      0.000      -3.292      -2.510
    x8            -0.1527      0.032     -4.712      0.000      -0.216      -0.089
    x9            -0.3176      0.038     -8.314      0.000      -0.392      -0.243
    ==============================================================================
            Logit Marginal Effects       
    =====================================
    Dep. Variable:                 target
    Method:                          dydx
    At:                           overall
    ==============================================================================
                    dy/dx    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.1403      0.006     23.352      0.000       0.129       0.152
    x2             0.0021      0.005      0.413      0.680      -0.008       0.012
    x3             0.1302      0.011     11.608      0.000       0.108       0.152
    x4             0.0151      0.005      2.958      0.003       0.005       0.025
    x5            -0.0049      0.005     -1.015      0.310      -0.014       0.005
    x6             0.0055      0.007      0.769      0.442      -0.008       0.019
    x7            -0.4598      0.030    -15.393      0.000      -0.518      -0.401
    x8            -0.0242      0.005     -4.747      0.000      -0.034      -0.014
    x9            -0.0503      0.006     -8.498      0.000      -0.062      -0.039
    ==============================================================================
    


```python
from sklearn.metrics import accuracy_score
pred = logit.predict()
#accuracy_score(np.where(pred >= 0.5, 1, 0), y)
```


```python
from sklearn.metrics import classification_report
display(HTML(f'''<h3 style="text-align:left">Table 9. Classification Report</h3>'''))
print(classification_report(y,np.where(pred >= 0.5, 1, 0)))
```


<h3 style="text-align:left">Table 9. Classification Report</h3>


                  precision    recall  f1-score   support
    
               0       0.87      0.66      0.75      3199
               1       0.72      0.90      0.80      3199
    
        accuracy                           0.78      6398
       macro avg       0.79      0.78      0.77      6398
    weighted avg       0.79      0.78      0.77      6398
    
    

The classification report for the resulting logistic regression model given by Table 9 has an overall accuracy of 78 percent. If we take hit songs as the relevant case for the dataset, the precision implies that the model can accurately identify hit songs by 72 percent. While recall of 0.90 suggests that the model can find 90 percent of hit songs in the data.

<p style="text-align:justify">From the results of the logistic regression, the features with the highest marginal effects (in absolute value) are <b><i>valence, danceability, and instrumentalness</i></b>.  This shows that valence and danceability provide the most positive impact in songs getting closer to a hit chart. Meanwhile, instrumentalness provides the most negative impact in songs, limiting the chances of landing on the hit charts.  Key, speechinees and acousticness were found to not significantly impact the likelihood of being a hit song. Though other features also have marginal effects, their magnitude is considerable lower than the top 3 so we can consider the impact of these features to be insubstantial.

Looking closer at the marginal effects, instrumentalness outweighs the combined impact of valence and danceability.  Therefore, for a song to reach the hit charts, the feature score of both valence and danceability should exceed the feature score of instrumentalness by at least twice to surpass the negative impact.

## V. Conclusion

<p style="text-align:justify">
Using the 2010-2019 Spotify Hits Predictor Dataset from Kaggle, we attempt to reveal the combination of audio features that determine hit songs from flops using a logistic regression.
<br><br>Looking into the general composition of audio features between hits and flops, we visually confirm differences in the means and distributions between the two groups. To test the statistical significance of these differences, we use Mann-Whitney U test for means since our features were found to be non-normally distributed.
<br><br>
After statistically confirming the differences between the means of the two groups, we then assessed the validity of the model and its potential predictors by testing for multicollinearity and linearity with log-odds. Energy and time-related variables were dropped to meet the assumptions of a logistic regression to allow for valid interpretations of the resulting summary statistics. After implementing a grid search over 10 trials and 10 splits, we then implemented an L1-regularized logistic regression using the C parameter with the highest test accuracy.
<br><br> The resulting model has an accuracy score of 78 percent. Summary statistics show that positive and danceable songs have higher chances of hitting the charts. Meanwhile, instrumentalness outweighs the combined impact of valence and danceability. So if the song is predicted to reach the hit charts, the feature score of both valence and danceability should exceed the feature score of instrumentalness by at least 2 times to surpass the negative impact. This suggests that a balance of certain features can contribute to a song's mass appeal and can land it at the top of the charts.
<br><br>This study can potentially help artists and music producers find commercial success by producing songs that fit the standard of those that land on the charts. Recording companies can also predict the potential success of songs they will be releasing based on the song's audio features.

</p>

## VI. Recommendations

<p style="text-align:justify">
In this report, we assumed that the mix of audio features plays the most important role in deciding whether a song will enter the weekly Billboard's Hot 100 list or not. However, it can be argued that in reality, song lyrics also have a significant impact on the commercial success of a song. Hence, for further studies on the topic, we recommend that datasets and analysis on song lyrics be included as well to provide a more solid conclusion. We also propose to extend the dataset to include songs from earlier decades and analyze the changes in the impact of different features over time. Moreover, the datasets of a more globalized version of the music industry standard be used in place of the Billboard chart. This will entail a more extensive study but in turn, will produce more generalized results.
Another recommendation for further studies is to look into local hit charts. This will allow for comparison among preferences locally and internationally. Furthermore, the study will also be made more relatable to the local audience.
</p>

## References

[1] Company, Houghton Mifflin Harcourt Publishing. "The American Heritage Dictionary entry: Music". ahdictionary.com. Retrieved 2021-01-20.

[2] Popular Music. (2015). Funk & Wagnalls New World Encyclopedia

[3] Reece, K. (2018). *Why Pop Music is Bad.* Medium. https://medium.com/@18kreece/why-pop-music-is-bad-67db0ae4cce2

[4] Python for Data Science. (n.d.). *Logistic Regression.* https://pythonfordatascienceorg.wordpress.com/logistic-regression-python/

[5] Chen, E. (2018). *Logistic Regression.* Blogspot. http://edchentech.blogspot.com/2018/06/logistic-regression.html?m=1

[6] Kaggle. Spotify Hits Predictor Dataset. https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset

[7] Spotify. Web API Reference. https://developer.spotify.com/documentation/web-api/reference/#object-audiofeaturesobject

[8] Billboard. (2012). *The Best of 2012: The Year In Music.* https://www.billboard.com/articles/news/1481472/the-best-of-2012-the-year-in-music

[9] Vanity Fair. (2019). *10 Songs That Explain the 2010s.* https://www.vanityfair.com/style/2019/12/songs-of-the-decade-2010s

[10] Wikipedia. Billboard Hot 100. https://en.wikipedia.org/wiki/Billboard_Hot_100

[11] Mishra, S. (2020) *Methods for Normality Test with Application in Python.* Towards Data Science. https://towardsdatascience.com/methods-for-normality-test-with-application-in-python-bb91b49ed0f5

[12] Mishra, P. et al. (2019). *Descriptive Statistics and Normality Tests for Statistical Data.* National Center for Biotechnology Information. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6350423/

[13] Tsagris, M. & Pandis, N. (2021) *Normality test: Is it really necessary?* American Journal of Orthodontics and Dentofacial Orthopedics. https://www.ajodo.org/article/S0889-5406(21)00038-X/fulltext

[14] Lewinson, S. (2019) *Explaining probability plots.* Towards Data Science. https://towardsdatascience.com/explaining-probability-plots-9e5c5d304703

[15] Varshney, P. (2020) *Q-Q Plots Explained.* Towards Data Science. https://towardsdatascience.com/q-q-plots-explained-5aa8495426c0

[16] Laerd Statistics. Mann-Whitney U Test using SPSS Statistics. https://statistics.laerd.com/spss-tutorials/mann-whitney-u-test-using-spss-statistics.php

[17] Choueiry, George (2020) *What is an Acceptable Value for VIF?* Quantifying Health. https://quantifyinghealth.com/vif-threshold/

[18] Python for Data Science. Logistic Regression. https://pythonfordatascienceorg.wordpress.com/logistic-regression-python/
