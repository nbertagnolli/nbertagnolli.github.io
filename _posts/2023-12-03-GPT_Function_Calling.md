links: [[LLMs]]
# Introduction

Function calling with LLMs is one of the neatest features in this space I've seen in a while. It allows you to orchestrate workflows with human language. Think about that for a second. Instead of painstakingly writing out your own control flow and managing all edge cases you can describe what you want done, provide some functions, and the LLM will take care of the rest (mostly).  I think this is pretty revolutionary!

I wanted to build a really simple example walking through how you could implement an agent using function calling from a set of methods that you've created.  There are some great tools out there like LangChain which can help with it, but honestly I find them a bit too bulky and hard to reason about for many applications that I work on. When building production LLM services it's really really important to manage your prompts and keep hallucinations under control. In my experience some of the frameworks that feel like magic in the beginning can become hard to handle and reason about as they grow. This is in part due to all of the prompting behind the scene that you don't directly see.  The purpose of this tutorial is to create a lightweight function calling agent mostly from scratch to show how easy it is to get started without a framework. In this post we will:

1. Create a set of functions the agent can choose from
2. Use a simple sentence transformer to 

# What should our agent do?
The first step is to figure out what we want our agent to do and construct prompts to guide it.
We want a simple system where we can specify some linear tasks in human language and have our agent make decisions about how to do them.  For this task, we'll stay very simple and just have the agent perform some mathematical operations in order.  In reality these functions could do whatever you want but to limit dependencies and make things clear we'll stay simple.

Here is the workflow we'd like our agent to follow.

```python
workflow = [
	"Please add 1 and 5",
	"please multiply 5 by the result",
	"please divide the result by 15",
	"Say duck this many times "
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

# Function Calling

Let's start by defining three mathematical operations in a python module called `utils.py`

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

These three functions do basically nothing but we can give our agent the ability to use them through function calling.  OpenAI has trained a few of their models to understand how to recognize when to use provided "tools" (functions) and how to structure input for them.  The OpenAI API takes in a JSON representation of the function and can use this to structure calls to this function.  If we were to convert `add_two_numbers` to function calling JSON it would look like this:

```json
{'name': 'add_two_numbers',
 'description': 'add_two_numbers(a: float, b: float) -> float - This function will add a and b together and return the result.',
 'parameters': {'title': 'add_two_numbersSchemaSchema',
  'type': 'object',
  'properties': {'a': {'title': 'A', 'type': 'number'},
   'b': {'title': 'B', 'type': 'number'}},
  'required': ['a', 'b']}}
```

We can generate this in a few different ways.  The easiest is to use a nice builtin utility from langchain which will automatically construct these inputs for you from a given function.

```python
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import tool
function_calling_def = format_tool_to_openai_function(tool(add_two_numbers))
```

This returns what we saw above.  It's very convenient and we'll use it in this post.  I have had it mess up on me sometimes particularly when using default arguments.  If you want more control you can define a `pydantic` model for your function and use the `schema()` method to get a complinat JSON output.

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
We are missing the description field but ChatGPT will still be able to use it. I generally just use the nice LangChain helper.

Now that we understand the basics of formatting these functions let's use one with ChatGPT.
To do that we'll need to pass in a list of these functions to the chat completion endpoint.

```python
messages = [{"role": "system", "content": ""},
            {"role": "user", "content": "Please add 1 and 5"}]
function = format_tool_to_openai_function(tool(add_two_numbers))
result = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=messages, functions=[function])
```

We've just added in the `functions` parameter to the chat completions endpoint and look at what we get!
```json
<OpenAIObject chat.completion id=chatcmpl-8Rr8fKQwyPX6jAusROXRrS9lcRfhF at 0x12f09b560> JSON: {
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1701649813,
  "model": "gpt-3.5-turbo-16k-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "add_two_numbers",
          "arguments": "{\n  \"a\": 1,\n  \"b\": 5\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 78,
    "completion_tokens": 23,
    "total_tokens": 101
  },
  "system_fingerprint": null
}
```

Notice how under `choices` our `finish_reason` is set to `function_call`. This means that the model is recommending that we call a function.  We can figure out which function it want's to call by looking at `result.choices[0].message.function_call.name` In this case it's `add_two_numbers` We can then get the arguments with `result.choices[0].message.function_call.arguments` which will be a JSON parsable object. Pretty neat! Let's call the function now!

```python
import json
f = globals()[result.choices[0].message.function_call.name]
f(**json.loads(result.choices[0].message.function_call.arguments))
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
{
  "id": "chatcmpl-8Rt6jiX06UfYp33N4XxSb9T3CmbXW",
  "object": "chat.completion",
  "created": 1701657381,
  "model": "gpt-3.5-turbo-16k-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "multiply_two_numbers",
          "arguments": "{\n  \"a\": 5,\n  \"b\": 6\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 374,
    "completion_tokens": 23,
    "total_tokens": 397
  },
  "system_fingerprint": null
}
```

We see that there is `choices.message.content` is null. This is because when we use the function calling version of ChatGPT it passes back a `function_call` instead of content.  We need something to help us parse the input based on whether it's a function call or not.

```python
def gpt_process_function_calling(gpt_response):
    # Check to see if the call terminated on a function call.
    finish_reason = gpt_response.choices[0].finish_reason
    # We check if we finished for an explicit function call or if we finished because of a long query
    # and gpt suggests a function call
    if finish_reason == "function_call":
        function_name = gpt_response.choices[0].message.function_call.name
        arguments = json.loads(gpt_response.choices[0].message.function_call.arguments) 
        func = globals()[function_name]
        return func(**arguments)
    else:
        # if not just pass the response through.
        return gpt_response.choices[0].message.content
```

In this function we check to see if GPT's response is a function call and if so call a function. If not, access the chat bot's content directly.


# function finding
The more functions you send the harder it can be for GPT to find the right one. You can prefilter the functions to send in on each workflow step.

```python
from sentence_transformers import SentenceTransformer, util
import inspect
import types

# Assign the closest two tools for each step in the workflow
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
function_registry = [func for func in dir(tools) if isinstance(getattr(tools, func), types.FunctionType)]
function_descriptions = [inspect.getsource(globals()[func]) for func in function_registry]
registry_embeddings = embedder.encode(function_descriptions, convert_to_tensor=True)

top_k = min(3, len(function_descriptions))
workflow_functions = []
for query in workflow:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, registry_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    workflow_functions.append([function_registry[i] for i in top_results.indices.tolist()])


outputs = []
output = ""
for instruction, funcs in zip(workflow, workflow_functions):
    functions = [format_tool_to_openai_function(tool(globals()[t])) for t in funcs]
    response = open_ai_generation("", instruction + f" {output}", functions=functions)
    output = gpt_process_function_calling(response)
    outputs.append(output)
```