---
layout: post
title: "Don't GPT Like a Fool Use a Tool!"
data: 2023-12-03
categories: jekyll update
---

<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
  <link rel="canonical" href="https://towardsdatascience.com/how-to-get-feature-importances-from-any-sklearn-pipeline-167a19f1214">

</head>

<figure class="half">
	<img src="/assets/dont_gpt_like_a_fool_use_a_tool.jpeg">
	<figcaption>Photo by author with the help of StableDiffusion-xl</figcaption>
</figure>


# Introduction

Function calling with LLMs is one of the neatest features in this space I've seen in a while. It allows you to orchestrate workflows with human language. Think about that for a second. Instead of painstakingly writing out your own control flow and managing all edge cases you can describe what you want done, provide some functions, and the LLM will take care of the rest (mostly).  I think this is pretty revolutionary!

I wanted to build a really simple example walking through how you could implement an agent using function calling from a set of methods that you've created.  There are some great tools out there like LangChain which can help with it, but honestly I find them a bit too bulky and hard to reason about for many applications that I work on. When building production LLM services it's really really important to manage your prompts and keep hallucinations under control. In my experience some of the frameworks that feel like magic in the beginning can become hard to handle and reason about as they grow. This is in part due to all of the prompting behind the scene that you don't directly see.  The purpose of this tutorial is to create a lightweight function calling agent mostly from scratch to show how easy it is to get started without a framework. In this post we will:

# What should our agent do?
The first step is to figure out what we want our agent to do and construct prompts to guide it.
We want a simple system where we can specify some linear tasks in human language and have our agent make decisions about how to do them.  For this task, we'll stay very simple and just have the agent perform some mathematical operations in order.  In reality these functions could do whatever you want but to limit dependencies and make things clear we'll stay simple.

Here is the workflow we'd like our agent to follow.

```python
workflow = [
	"Please add 1 and 5",
	"please multiply 5 by: ",
	"please divide the following number up by 15: ",
	"Say duck this many times please: "
]
```

The catch is, instead of having the LLM perform the action directly we'll want it to use function calling to make this possible.  Let's create a quick function to call the chat completions API and try out these prompts without function calling.

```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

outputs = []
output = ""
for instruction in workflow:
	messages = [{"role": "system", "content": ""},
                {"role": "user", "content": instruction + f" {output}"}]
    output = openai.ChatCompletion.create(model=model, messages=messages)
    outputs.append(output.choices[0].message.content)
```

We start by grabbing our  OpenAI Key.  We step through each instruction and call out to ChatGPT to follow our instruction directly. It works pretty well out of the box (minus the last instruction) and we get the following result:

```python
['1 + 5 = 6',
 'To multiply 5 by 6, the result would be 30.',
 'To divide the result of 30 by 15, the answer would be 2.',
 'duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck']
```

Our problem is simple so this makes sense. The interesting bit comes when we augment ChatGPT with tools. Let's give it some functions to work with and see if we can fix this ducking problem!

# Function Calling through Tools!

Let's start by defining three mathematical functions.

```python
def add_two_numbers(a: float, b: float) -> float:
    """This function will add a and b together and return the result."""
    return a + b

def multiply_two_numbers(a: float, b: float) -> float:
    """This function will multiply a by be and return the result."""
    return a * b

def divide_two_numbers(a: float, b: float) -> float:
    """This function will divide a by b and return the result."""
    return a / b
```

These three functions do basically nothing but we can give our agent the ability to use them through tool calling.  OpenAI has trained a few of their models to understand how to recognize when to use provided "tools" (functions) and how to structure input for them.  The OpenAI API takes in a JSON representation of the function and can use this to structure calls to this function.  If we were to convert `add_two_numbers` to function calling JSON it would look like this:

```json
{'type': 'function',
 'function': {'name': 'add_two_numbers',
  'description': 'add_two_numbers(a: float, b: float) -> float - This function will add a and b together and return the result.',
  'parameters': {'title': 'add_two_numbersSchemaSchema',
   'type': 'object',
   'properties': {'a': {'title': 'A', 'type': 'number'},
    'b': {'title': 'B', 'type': 'number'}},
   'required': ['a', 'b']}}}
```

We can generate this in a few different ways.  The easiest is to use a nice builtin utility from LangChain which will automatically construct these inputs for you from a given function.

```python
from langchain.tools.render import format_tool_to_openai_tool
from langchain.agents import tool
function_calling_def = format_tool_to_openai_tool(tool(add_two_numbers))
```

This returns what we saw above.  It's very convenient and we'll use it in this post.  I have had it mess up on me sometimes particularly when using default arguments.  If you want more control you can define a `pydantic` model for your function and use the `schema()` method to get a compliant JSON output.

```python
from pydantic import BaseModel

class AddTwoNumbers(BaseModel):
    a: float
    b: float

AddTwoNumbers.schema()
```

Running this yields:
```json
{'title': 'AddTwoNumbers',
 'type': 'object',
 'properties': {'a': {'title': 'A', 'type': 'number'},
  'b': {'title': 'B', 'type': 'number'}},
 'required': ['a', 'b']}
```
We are missing the description field and the upper type field. ChatGPT will still be able to use it without a description but we need to manually add in the type. I generally just use the nice LangChain helper.

Now that we understand the basics of formatting these functions let's use one with ChatGPT.
To do that we'll need to pass in a list of these functions to the chat completion endpoint.

```python
messages = [{"role": "system", "content": ""},
            {"role": "user", "content": "Please add 1 and 5"}]
function = format_tool_to_openai_function(tool(add_two_numbers))
result = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=messages, tools=[function])
```

We've just added in the `functions` parameter to the chat completions endpoint and look at what we get!
```json
{'id': 'chatcmpl-8SGersCGYmrdnIu67Or81Jhg2iNAS',
 'choices': [{'finish_reason': 'tool_calls',
   'index': 0,
   'message': {'content': None,
    'role': 'assistant',
    'function_call': None,
    'tool_calls': [{'id': 'call_bxEckzbIygLcQtixYHScBfCo',
      'function': {'arguments': '{\n  "a": 1,\n  "b": 5\n}',
       'name': 'add_two_numbers'},
      'type': 'function'}]}}],
 'created': 1701747909,
 'model': 'gpt-3.5-turbo-16k-0613',
 'object': 'chat.completion',
 'system_fingerprint': None,
 'usage': {'completion_tokens': 23, 'prompt_tokens': 78, 'total_tokens': 101}}
```

Notice how under `choices` our `finish_reason` is set to `tool_calls`. This means that the model is recommending that we call a tool.  We can figure out which tool it want's to call by looking at `result.choices[0].message.tool_calls[0].name` In this case it's `add_two_numbers` We can then get the arguments with `result.choices[0].message.tool_calls[0].arguments` which will be a JSON parsable object. Pretty neat! Let's call the function now!

```python
import json
f = globals()[result.choices[0].message.tool_calls[0].name]
f(**json.loads(result.choices[0].message.tool_calls[0].arguments))
```

And we get 6!  Here we take advantage of python's `globals()` function.   `globals()` is a built-in function that returns a dictionary containing the current global symbol table. The global symbol table is a namespace that contains all the global variables, functions, and other objects defined at the top level of a script or module. When you define a variable or function outside of any function or class in your Python code, it becomes a part of the global symbol table.  This allows us to use the string name of a function to grab the function directly and call it.

# Calling all Functions

This worked great on the one function! But the real power comes in an LLM's ability to figure out from the language of the code itself which function to call. Let's follow our whole workflow through with all of the functions we made and see what happens.


```python
funcs = ["add_two_numbers", "multiply_two_numbers", "divide_two_numbers"]
functions = [format_tool_to_openai_function(tool(globals()[t])) for t in funcs]

outputs = []
output = ""
for instruction in workflow:
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": instruction + f" {output}"}]
    output = openai.ChatCompletion.create(model=GPT_MODEL_NAME, messages=messages, functions=functions)
    outputs.append(output.choices[0].message.content)
```

Running this yields:
```python
[None, None, None, None]
```

Curious why is that? If we print out `output` we can get an idea about what's happening here.

```json
{'id': 'chatcmpl-8SGki2ypSTmdwMo0WtYjKrM9AYsb7',
 'choices': [{'finish_reason': 'tool_calls',
   'index': 0,
   'message': {'content': None,
    'role': 'assistant',
    'function_call': None,
    'tool_calls': [{'id': 'call_V5kcyVWTeIQMHxzUGsoYS8hy',
      'function': {'arguments': '{\n  "a": 5,\n  "b": 15\n}',
       'name': 'multiply_two_numbers'},
      'type': 'function'}]}}],
 'created': 1701748272,
 'model': 'gpt-3.5-turbo-16k-0613',
 'object': 'chat.completion',
 'system_fingerprint': None,
 'usage': {'completion_tokens': 23, 'prompt_tokens': 352, 'total_tokens': 375}}
```

We see that there is `choices.message.content` is null. This is because when we use the function calling version of ChatGPT it passes back a `tool_calls` instead of content.  We need something to help us parse the input based on whether it's a function call or not.

```python
def gpt_process_function_calling(gpt_response):
    # Check to see if the call terminated on a function call.
    finish_reason = gpt_response.choices[0].finish_reason
    # We check if we finished for an explicit function call or if we finished because of a long query
    # and gpt suggests a function call
    if finish_reason == "tool_calls":
        function_name = gpt_response.choices[0].message.tool_calls[0].function.name
        arguments = json.loads(gpt_response.choices[0].message.tool_calls[0].function.arguments)
        func = globals()[function_name]
        return func(**arguments)
    else:
        # if not just pass the response through.
        return gpt_response.choices[0].message.content
```

In this function we check to see if GPT's response is a function call and if so call a function. If not, access the chat bot's content directly.  This allows us to mix function calling with normal GPT prompting as in the final step of our workflow which doesn't have a function to help.  Now if we change output to be:

```python
raw_output = openai.chat.completions.create(model=GPT_MODEL_NAME, messages=messages, functions=functions)
output = gpt_process_function_calling(raw_output)
```
We get what we were after!!!
```
[6, 30, 2.0, 'duck duck']
```
GPT made a decision to reason itself instead of make the tool call! This is really pretty incredible. However, as I was working on this blog post this result was intermittent. 50% of the time it would choose to make a function call. This is both annoying and it highlights something important. These systems are stochastic in that there is some random in how they behave. If you want to ensure the correct behavior you might consider doing a little engineering around which functions get passed as tools.  In the next section we'll discuss a very basic system for limiting tool context on GPT calls.

# function finding
We need to figure out if a function is a good fit for a command in our workflow.  This has the added benefit of letting us reduce the context sent to GPT as well.  The more functions you send the harder it can be for GPT to find the right one. You can prefilter the functions to send in on each workflow step using sentence-transformers!

```python
from sentence_transformers import SentenceTransformer, util
import inspect
import types
import torch

# Assign the closest two tools for each step in the workflow
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# List of all functions
function_registry = ["add_two_numbers", "multiply_two_numbers", "divide_two_numbers"]

# Use the inspect library to get a string version of the function
# including the docstring
function_descriptions = [inspect.getsource(globals()[func]) for func in function_registry]

# Embed every function
function_embeddings = embedder.encode(function_descriptions, convert_to_tensor=True)

top_k = min(1, len(function_descriptions))
workflow_functions = []

# step through each workflow instruction and encode it.
for query in workflow:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, function_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
	# We only add a function if it's closer than .25 by cosine distance.
    if max(cos_scores) > .2:
        workflow_functions.append([function_registry[i] for i in top_results.indices.tolist()])
    else:
        workflow_functions.append([])
```

The output of this has a single function for each step:

```
[['add_two_numbers'],
 ['multiply_two_numbers'],
 ['divide_two_numbers'],
 []]
```

Notice we use a .2 cutoff for the cosine similarity saying that any instruction which has less than this score for all similarities shouldn't have any function calls.  The last step is adding in some argument filtering based on whether or not we think a function should be used.
```python
outputs = []
output = ""
for instruction, functions in zip(workflow, workflow_functions):
    kwargs = {}
    if len(functions) > 0:
        functions = [format_tool_to_openai_tool(tool(globals()[t])) for t in functions]
        kwargs = {"tools": functions}
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": instruction + f" {output}"}]
    print(messages)
    output = openai.chat.completions.create(model=GPT_MODEL_NAME,
                                            messages=messages,
                                            **kwargs)
    output = gpt_process_function_calling(output)
    outputs.append(output)
```

And now we get two `duck duck` every time!!!

# Conclusion
Tool/Function calling is a game changer in the LLM space. This allows us to use LLMs to orchestrate workflows with human language. This is incredibly powerful. Think about how UIs will change under this paradigm, or how users will be able to interact with APIs without technical know how. Take the knowledge we acquired here and go build something cool!  A full Jupyter notebook can be found [here](https://gist.github.com/nbertagnolli/016badab109b46b9510206cf5e6e67c0).