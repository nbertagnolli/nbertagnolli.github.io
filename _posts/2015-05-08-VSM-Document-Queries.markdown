---
layout: post
title: "VSM's a Brief Tutorial on Document Retrieval"
date: 2015-07-31
categories: jekyll update
---

<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>

#Introduction
Recently I've been playing around a lot with vector space models (VSM) for words and this has led me to learn a lot about the history of these things.  I found that one of the earliest successful applications of a VSM was in document retrieval and I thought that I'd try to throw together a quick tutorial on the matter.  In this tutorial I will:

* Motivate the use of VSMs in linguistics
* Construct a few different rudimentary VSMs
* Perform document retrieval using these VSMs

#What are VSMs in Linguistics?
Simple, a VSM in linguistics is constructed by representing a certain linguistic element word, document, etc. as a vector.  That's it.  Now one might ask, “why is this useful?”  In linguistics the study of <a target = "_blank" href = "https://en.wikipedia.org/wiki/Lexical_semantics">Lexical Semantics</a> seeks to leverage lexical relationships to understand overall sentence meaning. If we want to examine relationships amongst lexical elements, like words, a vector space seems like a natural place to do this due to the simplicity of assessing distance and similarity.  Think Euclidean distance and cosine similarity. Let's construct the most rudimentary VSM for words.  To do this we represent each word as a one hot encoding in a vector the size of the vocabulary.  Let’s look at what the words dog, hound, and boat look like under this construction.
\begin{align}dog &= [\begin{matrix} … & 0 & 0 & 1 & 0 & 0 & …\end{matrix}]\\\ hound &= [\begin{matrix} … & 0 & 1 &0 & 0 & 0 & ...\end{matrix}]\\\ boat &= [\begin{matrix}… & 0 & 0 & 0 & 1 & 0 & …\end{matrix}]\end{align} 
We can now ask questions like how similar are boat and dog or dog and hound because they are represented by real vectors.  Though this is not particularly informative given that:
\\[dog \cdot hound = dog \cdot boat = hound\cdot boat = 0\\]
To truly gain anything from this vector space we’ll need to construct a better representation.  Hopefully, we can find a representation where lexical elements like $$dog$$ and $$hound$$ are similar but $$dog$$/$$hound$$ and $$boat$$ are not.

#Term-Document Matrix
To do this we begin with the study of documents. Create a space where the dimension of our vectors is equivalent to the number of words in our vocabulary.  Stated formally as \\(d \in \mathbb{R}^{|V|}\\) where $$d$$ is a document vector and $$|V|$$ is the number of words in our vocabulary.  We now construct a matrix where each row represents a word, each column is a document, and each entry in the matrix is the number of times that word $$i$$ appears in document $$j$$.  Let’s illustrate this with a simple example.  Take for example the four documents below:
\begin{align} d_1 &= \text{the cat,the dog and the monkey swam}\\\ d_2 &= \text{a cat sat}\\\ d_3 &= \text{the dog sat}\\\ d_4 &= \text{the monkey swam}\end{align} For this set of documents the term-document matrix would be:
\begin{align} D &= \begin{matrix}the\\\ cat\\\ dog\\\ monkey\\\ swam\\\ sat\\\ and\\\ a\end{matrix}\left[\begin{matrix}3 & 0 & 1 & 1\\\ 1 & 1 & 0 & 0\\\ 1 & 0 & 1 & 0\\\ 1 & 0 & 0 & 1\\\ 1 & 0 & 0 & 1\\\ 0 & 1 & 1 & 0\\\ 1 & 0 & 0 & 0\\\ 0 & 1 & 0 & 0\end{matrix}\right]\end{align}
We can now try and assess document similarity by examining the dot product of the column vectors associated with the relevant documents.  Moreover, we can find documents that are closest to a query.  For example, if I wanted to find the document from our set closest to my query of: \\\[q = \text{the cat sat}\\\] I would find that $$d_1$$ is the most similar because:\begin{align}d_1 \cdot q &= [\begin{matrix}3 & 1 & 1 & 1 & 1 & 0 & 1 & 0\end{matrix}] \cdot [\begin{matrix} 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0\end{matrix}]\\\ &= 4\end{align} which is larger value than can be attained using any other document.  However, this is strange because $$d_1$$ does not talk about sitting, or just a cat.  $$d_2$$ should have been the right answer.  This is because our current system gives too much weight to words that appear frequently.  We can address this with the notion of TF-IDF

#TF-IDF
The most basic way to address this is to down weight words that appear in many documents.  This can be done using a method known as term frequency inverse document frequency (TF-IDF).  Here we take each element in our term document matrix as: \\[M_{i,j} = \sum_{w\in d_j}\mathbb{1}[w = w_i] \times \log \left[\frac{N}{n_i}\right]\\].  where: \begin{align} M_{ij} &= \text{Element in row i and column j of the term document matrix}\\\ w &= \text{Word in document }d_j\\\ w_i &= \text{Word corresponding to row }i\\\ N &= \text{Number of documents in our corpus}\\\ n_i &= \text{Number of documents that word } w_i\text{ appears in}\end{align}Under this paradigm our term document matrix becomes:\begin{align} D &= \begin{matrix}the\\\ cat\\\ dog\\\ monkey\\\ swam\\\ sat\\\ and\\\ a\end{matrix}\left[\begin{matrix}.863 & 0 & 1.3863 & 1.3863\\\ 1.3863 & 1.3863 & 0 & 0\\\ 1.3863 & 0 & 1.3863 & 0\\\ 1.3863 & 0 & 0 & 1.3863\\\ 1.3863 & 0 & 0 & 1.3863\\\ 0 & 1.3863 & 1.3863 & 0\\\ 1.3863 & 0 & 0 & 0\\\ 0 & 1.3863 & 0 & 0\end{matrix}\right]\end{align} Now we see that our query is most similar to $$d_2$$ as we would expect.  There is still one last problem; that the model will give preference to larger documents because their vectors will be larger.  This can be addressed by normalizing the matrix.  

#Future Thoughts
So we can answer basic document queries by:

1. Constructing a word document matrix<
2. Normalizing these vectors
3. Representing a query as a normalized vector in this space
4. Taking the dot product of the query and all document vectors
5. Ranking documents based on highest similarty



<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-68394304-1', 'auto');
  ga('send', 'pageview');

</script>










