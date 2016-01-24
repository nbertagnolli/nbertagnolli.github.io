---
layout: post
title: "Covariance and Visual Normality"
data: 2015-01-21
categories: jekyll update
---

<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>


## **Introduction**
    Recently I've been feeling the need to delve into statistics a bit more than I have in the past.  I've always felt that a strong theoretical understanding of mathematics would be sufficient to excel at statistics.  Unfortunately, I have come to realize that there is a great wealth of information in experience.  I love playing with data, and teasing meaning from chaos.  I plan to use the next few posts to begin looking at practical statistics in a bit more depth.  This particular post steps through the basics of chapter one of Robert Gentleman, Kurt Hornik, and Giovanni Parmigiani's book Multivariate analysis in R. Except here we are using Python, because I like python better.  The first chapter concerns itself mostly with the concept of covariance.  We will address the simple issue of calculating covariance and correlation, and then use these concepts to visually determine if a dataset is normally distributed.
    
    For this discussion I will be using the abalone dataset found here http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/.  All analysis will be done in python.


## **Covariance**

Covariance is a way of assessing how much two random variables depend on one another.  It can be used to give us an idea of how the change in one variable could potentially affect the change in another.  Covariance is defined as:
\begin{align*}
  \text{Cov}(X_i, X_j) = \mathbb{E}(X_i-\mu_i)(X_j - \mu_j)
\end{align*}

Where $$\mu_i = \mathbb{E}(X_i)$$ and $$\mu_j = \mathbb{E}(X_j)$$

But there are many many ways to calculate covaraince which can be derived from the above equation such as:

\begin{align*}
\text{Cov}(X,X) = \frac{1}{n}\sum_{i=1}^n (x_i-\bar{X_i})(x_i-\bar{X_i})^T\\
\end{align*}

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

So far so good.  Now let's use pandas to calculate the covariance of the data.

{% highlight python %}
# To find covariance remove the categorical attribute gender
names = abalone.columns
abalone[names[0:]].cov()  # Remove the categorical attribute sex
{% endhighlight %}

Something interesting to note is that the variance of each attribute lies on the diagonal of the covariance matrix.  This is all fine and good but there is a real problem here.  These covariances are hard to interpret because of the difference in scale between each of the features.  Take for example the difference between length and rings.  Rings has a maximum of 29 and length has a maximum of .8!  A better way to examine these results is to look at what is known as correlation.

## **Correlation**
Here we deal with the issue of scale by normalizing based on the standard deviation of each variable.  The elements in our new matrix will be:

\begin{align*}
\rho_{ij} &= \frac{\sigma_{ij}}{\sigma_i\sigma_j}
\end{align*}

Correlation is much easier to interpret in that it describes a linear relationship between the variables of interest.  If it is positive and large it indicates that if variable 1 is large then we might be able to expect variable 2 being large as well and vice-versa.

Let's see what this looks like for our abalone dataset:

{% highlight python %}
abalone[names[0:]].corr()  # Remove the categorical attribute sex
{% endhighlight %}

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

Would you look at that, almost perfectly linear!  This shows that our points were most likely drawn from a normal distribution so our sanity check passed! 

Now let's try this with the height feature of the abalone data.  

{% highlight python %}
# Create a quantile-quantile plot of normal data  
stats.probplot(abalone["height", dist="norm", plot=plt)
plt.show()
{% endhighlight %}

The curve is exceptionally linear! Except for a few troublesome outliers.

What about the rings feature:

This curve deviates quite a bit from linear so I would be hesitant to conclude that the abalone's rings are normally distributed.

What about the whole abalone?  Are the features collectively drawn from a multivariate normal distribution?  This may appear like a much trickier question but in actuality it is quite easy to determine.  All we need to do is examine the Mahalanobis distance between the points and then compare to the  $$\chi^2_q$$ distribution.  If we do that we get:























