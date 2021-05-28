---
layout: post
title: "One Weird Trick to Ace any Dynamic Programming Interview"
data: 2021-01-11
categories: jekyll update
---

<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
  <link rel="canonical" href="https://towardsdatascience.com/how-to-get-feature-importances-from-any-sklearn-pipeline-167a19f1214">

</head>

<figure class="half">
	<img src="/assets/2021-01-12-one_weird_trick/title_image.jpeg">
	<figcaption>Photo by C. Cagnin from Pexels</figcaption>
</figure>

## **Introduction**

The technical interview is a cultural phenomenon that has penetrated to the core of almost any tech company these days. It seems almost impossible to get a software job without having to answer some arcane question from a second-year algorithms and data structures course.

When I first started interviewing I remember having the most trouble with Dynamic Programming problems. You know, how to find a maximum sublist, or figure out if you can make change given a set of coins and an angry customer. Those really useful problems that I totally use all the time in my day to day job... Recently, I was in an interview and thought to myself, “this is dumb what is it actually testing? Why are they making me do this?” So I chose to have a little fun with the problem. In this tutorial, I’m going to walk through a somewhat tongue-in-cheek way of memoizing any recursive solution to a problem. This will allow you to learn one technique and be able to memoize any recursive function with immutable inputs. Yes, you heard me, ANY recursive function with immutable inputs.

Let’s get started! We’ll take a look at the subset-sum problem.

## **Appeal to Reader**
If you pay for Medium, or haven't used your free articles for this month, please consider reading <a href="https://towardsdatascience.com/one-weird-trick-to-ace-any-dynamic-programming-interview-4584096a3f9f">this article there</a>.  I post all of my articles here for free so everyone can access them, but I also like beer and Medium is a good way to collect some beer money : ). So please consider buying me a beer by reading this article on Medium.

## **Subset-Sum Problem**

Given a list of positive integers ls=(1..n) and an integer k, is there some subset of ls that sums to exactly k?

## **Initial Solution**

Start by constructing a simple recursive solution to this problem and don’t worry about the time complexity. Let’s think for a second, the easiest thing to do is to construct all possible sublists and take the sum of each of them. If the sum of a sublist is equal to our desired value k then we’re done and we return True else False. Seems pretty simple right? We can code that up in only a few lines below.

{% highlight python %}
def subset_sum(ls: Tuple[int], total: int, total_sum: int) -> bool:

    if total == total_sum:
        return True

    for idx, val in enumerate(ls):

        new_ls = ls[:idx] + ls[idx + 1:],
        if subset_sum(new_ls, total, val + total_sum):
            return True

    return False
{% endhighlight %}

Establish a base case to see if the total_sum is equal to the desired total . That will tell us if we’ve hit our target and we should return True. With the base cases out of the way, we can focus on the main logic. It’s also fairly simple. We just need to step through each element in our list and:

1. Create a new list with our current index removed

2. Add our current value to our `total_sum`

3. Recurse down all the subsets from our new list.

Tada! we’ve solved this problem and proved we can practice interview problems to our new prospective company! They’ll surely hire us now!

But wait! The astute interviewee will note that this algorithm is terrible in terms of time complexity it’ll run in O(n!) 🙀. No one will hire us with such terrible performance. But we’re smart and we know that this problem is a dynamic programming problem so all we have to do is memoize it. To quote Wikipedia:

"Memoization is an optimization technique used primarily to speed up computer programs by storing the results of expensive function calls and returning the cached result when the same inputs occur again."

So the real nugget is to cache executions that we’ve run. What this means for example is when we’re executing subset_sum we need to remember the results from computations. Say we ran the function where the inputs were:

```
total_sum = 5
total = 10
ls = (3)
```

Our function adds 3 to 5 and yields 8 and our list is empty so we don’t reach 10 and return False. Memoization is just remembering that for the inputs of 5, 10, and (3) we get false so instead of doing the calculation we can just look up our cached value of False.

Python’s `functools` module has a decorator that does just that! It will automatically remember what the function returned if it has seen that input combination before. We can memoize our entire function with one line of python…

{% highlight python %}
from functools import lru_cache

@lru_cache(None)
def subset_sum(ls: Tuple[int], total: int, total_sum: int) -> bool:
   # All the previous code.
{% endhighlight %}

That’s it. We’re done. We memoized it and have a fully dynamic solution. Now, this is probably not what the interviewer wants. They're not interested in solutions to these problems they’re interested in watching you code for some reason 🤷‍♀. So our hypothetical interviewer retorts, “Very funny but let’s actually memoize this solution. Assume you don’t have the lru_cache function.”

To which you reply, “Okay good Madam/Sir I’m happy to oblige. How about we just implement lru_cache then? That way our memoization can be applied to any problem anyone at your organization might face. The solution will be superior to just memoizing this problem because it works in general.”

The interviewer gives a chuckle and agrees to your proposal taking you for a fool thinking there is no way you’ll be able to implement such a fancy pants function in the time remaining. But you’re not playing, you’re a badass. 💪

Let’s start simple. We need some way of knowing which function we are memoizing and we’ll also need some way of storing the previous function runs we’ve seen. We start by making a class with two initialized variables:

{% highlight python %}
class Memoize:

    def __init__(self, f):
        self.f = f
        self.memoized = {}
{% endhighlight %}

`f` stores the function we’re going to execute in our case `subset_sum`, `memoized` is a dictionary where the key will be the arguments to our function and the value will be the returned value of the function given those arguments. Now you see why the inputs to the function need to be immutable because we are hashing them in a dictionary.

We’re halfway there… Now we want this to operate like a decorator and wrap any function so we’ll add a `__call__` method. This method should take an arbitrary set of arguments so it’ll need to take `*args` as its input. Then the logic falls right out if we’ve never seen that set of args in our memorized dictionary execute the function with those arguments. Otherwise, we just return the cached value stored in our dictionary.

{% highlight python %}
class Memoize:

    def __init__(self, f):
        self.f = f
        self.memoized = {}

    def __call__(self, *args):
        if args not in self.memoized:
            self.memoized[args] = self.f(*args)
        return self.memoized[args]
{% endhighlight %}

That’s it…. We can now memoize our original function as:

{% highlight python %}
@Memoize
def subset_sum(ls: Tuple[int], total: int, total_sum: int) -> bool:
    # All the previous code.
{% endhighlight %}

The interviewer is stunned because they’ve never thought to solve memoization in general just in each specific case. You get the job and 10 minutes of your life back. It’s a win-win for everybody.

# Final Thoughts

One thing to keep in mind is this relies on the fact that you can create a hash of the inputs to your function. That’s why I’m using a Tuple instead of a List. If the object can’t be hashed it can’t be cached, at least in this case.