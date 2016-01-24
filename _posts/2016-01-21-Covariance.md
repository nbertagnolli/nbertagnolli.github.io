---
layout: post
title: "Covariance and Visual Normality"
data: 2016-01-21
categories: jekyll update
---

<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>


## **Introduction**



Recently I've been feeling the need to delve into statistics a bit more than I have in the past.  I've always felt that a strong theoretical understanding of mathematics would be sufficient to excel at statistics.  Unfortunately, I have come to realize that there is a great wealth of information in experience.  I love playing with data, and teasing meaning from chaos.  I plan to use the next few posts to begin looking at practical statistics in a bit more depth.  This particular post steps through the basics of chapter one of Robert Gentleman, Kurt Hornik, and Giovanni Parmigiani's book Multivariate analysis in R. Except here we are using Python, because I like python better.  The first chapter concerns itself mostly with the concept of covariance.  We will address the simple issue of calculating covariance and correlation, and then use these concepts to visually determine if a dataset is normally distributed.
    
For this discussion I will be using the abalone dataset found  <a target = "_blank" href = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/">here</a>.  All analysis will be done in python.


## **Covariance**

Covariance is a way of assessing how much two random variables depend on one another.  It can be used to give us an idea of how the change in one variable could potentially affect the change in another.  Covariance is defined as:
\begin{align}
  \text{Cov}(X_i, X_j) = \mathbb{E}(X_i-\mu_i)(X_j - \mu_j)
\end{align}

Where $$\mu_i = \mathbb{E}(X_i)$$ and $$\mu_j = \mathbb{E}(X_j)$$

But there are many many ways to calculate covaraince which can be derived from the above equation such as:

\begin{align}
\text{Cov}(X,X) = \frac{1}{n}\sum_{i=1}^n (x_i-\bar{X_i})(x_i-\bar{X_i})^T\\
\end{align}

Where$$X_i$$ are the rows of $$X$$.  This can be expressed more precisely as: \[X^TX\].

To begin Let's take a look at our data using pandas.  We load in the csv and then display the first five rows of our dataset.
{% highlight python %}
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Read in the CSV of the measure data
abalone = pd.read_csv("/Users/tetracycline/Data/abalone.csv")
abalone.head()
{% endhighlight %}

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>length</th>
      <th>diameter</th>
      <th>height</th>
      <th>whole_weight</th>
      <th>shucked_weight</th>
      <th>viscera_weight</th>
      <th>shell_weight</th>
      <th>rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>0.455</td>
      <td>0.365</td>
      <td>0.095</td>
      <td>0.5140</td>
      <td>0.2245</td>
      <td>0.1010</td>
      <td>0.150</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>0.350</td>
      <td>0.265</td>
      <td>0.090</td>
      <td>0.2255</td>
      <td>0.0995</td>
      <td>0.0485</td>
      <td>0.070</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>0.530</td>
      <td>0.420</td>
      <td>0.135</td>
      <td>0.6770</td>
      <td>0.2565</td>
      <td>0.1415</td>
      <td>0.210</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.440</td>
      <td>0.365</td>
      <td>0.125</td>
      <td>0.5160</td>
      <td>0.2155</td>
      <td>0.1140</td>
      <td>0.155</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I</td>
      <td>0.330</td>
      <td>0.255</td>
      <td>0.080</td>
      <td>0.2050</td>
      <td>0.0895</td>
      <td>0.0395</td>
      <td>0.055</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


So far so good.  Now let's use pandas to calculate the covariance of the data.

{% highlight python %}
# To find covariance remove the categorical attribute gender
names = abalone.columns
abalone[names[0:]].cov()  # Remove the categorical attribute sex
{% endhighlight %}



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length</th>
      <th>diameter</th>
      <th>height</th>
      <th>whole_weight</th>
      <th>shucked_weight</th>
      <th>viscera_weight</th>
      <th>shell_weight</th>
      <th>rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>length</th>
      <td>0.014422</td>
      <td>0.011761</td>
      <td>0.004157</td>
      <td>0.054491</td>
      <td>0.023935</td>
      <td>0.011887</td>
      <td>0.015007</td>
      <td>0.215562</td>
    </tr>
    <tr>
      <th>diameter</th>
      <td>0.011761</td>
      <td>0.009849</td>
      <td>0.003461</td>
      <td>0.045038</td>
      <td>0.019674</td>
      <td>0.009787</td>
      <td>0.012507</td>
      <td>0.183872</td>
    </tr>
    <tr>
      <th>height</th>
      <td>0.004157</td>
      <td>0.003461</td>
      <td>0.001750</td>
      <td>0.016803</td>
      <td>0.007195</td>
      <td>0.003660</td>
      <td>0.004759</td>
      <td>0.075179</td>
    </tr>
    <tr>
      <th>whole_weight</th>
      <td>0.054491</td>
      <td>0.045038</td>
      <td>0.016803</td>
      <td>0.240481</td>
      <td>0.105518</td>
      <td>0.051946</td>
      <td>0.065216</td>
      <td>0.854409</td>
    </tr>
    <tr>
      <th>shucked_weight</th>
      <td>0.023935</td>
      <td>0.019674</td>
      <td>0.007195</td>
      <td>0.105518</td>
      <td>0.049268</td>
      <td>0.022675</td>
      <td>0.027271</td>
      <td>0.301204</td>
    </tr>
    <tr>
      <th>viscera_weight</th>
      <td>0.011887</td>
      <td>0.009787</td>
      <td>0.003660</td>
      <td>0.051946</td>
      <td>0.022675</td>
      <td>0.012015</td>
      <td>0.013850</td>
      <td>0.178057</td>
    </tr>
    <tr>
      <th>shell_weight</th>
      <td>0.015007</td>
      <td>0.012507</td>
      <td>0.004759</td>
      <td>0.065216</td>
      <td>0.027271</td>
      <td>0.013850</td>
      <td>0.019377</td>
      <td>0.281663</td>
    </tr>
    <tr>
      <th>rings</th>
      <td>0.215562</td>
      <td>0.183872</td>
      <td>0.075179</td>
      <td>0.854409</td>
      <td>0.301204</td>
      <td>0.178057</td>
      <td>0.281663</td>
      <td>10.395266</td>
    </tr>
  </tbody>
</table>
</div>


Something interesting to note is that the variance of each attribute lies on the diagonal of the covariance matrix.  This is all fine and good but there is a real problem here.  These covariances are hard to interpret because of the difference in scale between each of the features.  Take for example the difference between length and rings.  Rings has a maximum of 29 and length has a maximum of .8!  A better way to examine these results is to look at what is known as correlation.

## **Correlation**
Here we deal with the issue of scale by normalizing based on the standard deviation of each variable.  The elements in our new matrix will be:

\begin{align}
\rho_{ij} &= \frac{\sigma_{ij}}{\sigma_i\sigma_j}
\end{align}

Correlation is much easier to interpret in that it describes a linear relationship between the variables of interest.  If it is positive and large it indicates that if variable 1 is large then we might be able to expect variable 2 being large as well and vice-versa.

Let's see what this looks like for our abalone dataset:

{% highlight python %}
abalone[names[0:]].corr()  # Remove the categorical attribute sex
{% endhighlight %}


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length</th>
      <th>diameter</th>
      <th>height</th>
      <th>whole_weight</th>
      <th>shucked_weight</th>
      <th>viscera_weight</th>
      <th>shell_weight</th>
      <th>rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>length</th>
      <td>1.000000</td>
      <td>0.986812</td>
      <td>0.827554</td>
      <td>0.925261</td>
      <td>0.897914</td>
      <td>0.903018</td>
      <td>0.897706</td>
      <td>0.556720</td>
    </tr>
    <tr>
      <th>diameter</th>
      <td>0.986812</td>
      <td>1.000000</td>
      <td>0.833684</td>
      <td>0.925452</td>
      <td>0.893162</td>
      <td>0.899724</td>
      <td>0.905330</td>
      <td>0.574660</td>
    </tr>
    <tr>
      <th>height</th>
      <td>0.827554</td>
      <td>0.833684</td>
      <td>1.000000</td>
      <td>0.819221</td>
      <td>0.774972</td>
      <td>0.798319</td>
      <td>0.817338</td>
      <td>0.557467</td>
    </tr>
    <tr>
      <th>whole_weight</th>
      <td>0.925261</td>
      <td>0.925452</td>
      <td>0.819221</td>
      <td>1.000000</td>
      <td>0.969405</td>
      <td>0.966375</td>
      <td>0.955355</td>
      <td>0.540390</td>
    </tr>
    <tr>
      <th>shucked_weight</th>
      <td>0.897914</td>
      <td>0.893162</td>
      <td>0.774972</td>
      <td>0.969405</td>
      <td>1.000000</td>
      <td>0.931961</td>
      <td>0.882617</td>
      <td>0.420884</td>
    </tr>
    <tr>
      <th>viscera_weight</th>
      <td>0.903018</td>
      <td>0.899724</td>
      <td>0.798319</td>
      <td>0.966375</td>
      <td>0.931961</td>
      <td>1.000000</td>
      <td>0.907656</td>
      <td>0.503819</td>
    </tr>
    <tr>
      <th>shell_weight</th>
      <td>0.897706</td>
      <td>0.905330</td>
      <td>0.817338</td>
      <td>0.955355</td>
      <td>0.882617</td>
      <td>0.907656</td>
      <td>1.000000</td>
      <td>0.627574</td>
    </tr>
    <tr>
      <th>rings</th>
      <td>0.556720</td>
      <td>0.574660</td>
      <td>0.557467</td>
      <td>0.540390</td>
      <td>0.420884</td>
      <td>0.503819</td>
      <td>0.627574</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

From the above table it appears that most things have a positive relationship meaning that if one of the features increases it could be expected that the other will be larger as well.  This makes sense given that most of the data is physical and size based.  Take a look at the diameter row.  Here wee see that the feature that has the most correlation with diameter is length.  This makes sense given that abalone are mostly round so length and diameter should behave very similarly.

# **Distribution Testing**
So this is all fine and good butcan we do something more interesting with this like determine if the features of the abalone are normally distributed?  To answer this question we will use the visual tool of a quantile-quantile plot.  This tool shows the 

For example let's see if points drawn from a normal distribution match a theoretical normal distribution.

{% highlight python %}
# Create a quantile-quantile plot of normal data 
normal_data = np.random.normal(loc = 20, scale = 5, size=100)   
stats.probplot(normal_data, dist="norm", plot=plt)
plt.show()
{% endhighlight %}

  <figure class="half">
	<img src="/assets/covariance_visual_normal/figure_01.png">
  </figure>

Would you look at that, almost perfectly linear!  This shows that our points were most likely drawn from a normal distribution so our sanity check passed! 

Now let's try this with the height feature of the abalone data.  

{% highlight python %}
# Create a quantile-quantile plot of normal data  
stats.probplot(abalone["height", dist="norm", plot=plt)
plt.show()
{% endhighlight %}

The curve is exceptionally linear! Except for a few troublesome outliers.

What about the rings feature:
  <figure class="half">
	<img src="/assets/covariance_visual_normal/figure_02.png">
  </figure>

This curve deviates quite a bit from linear so I would be hesitant to conclude that the abalone's rings are normally distributed.

What about the whole abalone?  Are the features collectively drawn from a multivariate normal distribution?  This may appear like a much trickier question but in actuality it is quite easy to determine.  All we need to do is examine the Mahalanobis distance between the points and then compare to the  $$\chi^2_q$$ distribution.  If we do that we get:

  <figure class="half">
	<img src="/assets/covariance_visual_normal/figure_03.png">
  </figure>






















