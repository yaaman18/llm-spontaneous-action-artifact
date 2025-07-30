# Python Language Specifications for Sub-Agents

## Overview
Python is a high-level, interpreted programming language that supports multiple programming paradigms. This document provides comprehensive language specifications to help sub-agents fully utilize Python's capabilities.

## Core Language Features

### 1. Data Model
- **Objects, Values, and Types**: Everything in Python is an object with identity, type, and value
- **Standard Type Hierarchy**: Numbers, sequences, mappings, classes, instances, exceptions
- **Special Method Names**: `__init__`, `__str__`, `__repr__`, `__len__`, `__getitem__`, etc.
- **Coroutines**: async/await syntax for asynchronous programming

### 2. Data Types

#### Built-in Types
- **Numeric**: `int`, `float`, `complex`
- **Sequence**: `str`, `list`, `tuple`, `range`
- **Set**: `set`, `frozenset`
- **Mapping**: `dict`
- **Boolean**: `bool` (subclass of int)
- **None**: `NoneType`

#### Advanced Types
- **Functions**: First-class objects with closures
- **Classes**: Object-oriented programming support
- **Modules**: Namespace containers
- **Generators**: Lazy evaluation with `yield`

### 3. Lexical Analysis

#### Identifiers and Keywords
```python
# Reserved keywords (cannot be used as identifiers)
False, None, True, and, as, assert, async, await, break, class, continue, 
def, del, elif, else, except, finally, for, from, global, if, import, 
in, is, lambda, nonlocal, not, or, pass, raise, return, try, while, with, yield
```

#### Literals
- **String literals**: `'single'`, `"double"`, `"""triple"""`, raw strings `r"raw"`
- **Numeric literals**: `42`, `3.14`, `1e10`, `0x1F`, `0o17`, `0b1010`
- **Boolean literals**: `True`, `False`

### 4. Operators and Expressions

#### Arithmetic Operators
```python
+, -, *, /, //, %, **  # Addition, subtraction, multiplication, division, floor division, modulo, exponentiation
```

#### Comparison Operators
```python
==, !=, <, <=, >, >=, is, is not, in, not in
```

#### Logical Operators
```python
and, or, not
```

#### Bitwise Operators
```python
&, |, ^, ~, <<, >>  # AND, OR, XOR, NOT, left shift, right shift
```

#### Assignment Operators
```python
=, +=, -=, *=, /=, //=, %=, **=, &=, |=, ^=, <<=, >>=
```

### 5. Control Flow Statements

#### Conditional Statements
```python
if condition:
    # code block
elif other_condition:
    # code block
else:
    # code block
```

#### Loops
```python
# For loop
for item in iterable:
    # code block

# While loop
while condition:
    # code block

# Loop control
break      # Exit loop
continue   # Skip to next iteration
```

#### Exception Handling
```python
try:
    # risky code
except SpecificException as e:
    # handle specific exception
except (Exception1, Exception2):
    # handle multiple exceptions
else:
    # runs if no exception
finally:
    # always runs
```

### 6. Functions and Classes

#### Function Definition
```python
def function_name(param1, param2=default, *args, **kwargs):
    """Docstring"""
    return value

# Lambda functions
lambda x, y: x + y
```

#### Class Definition
```python
class ClassName(BaseClass):
    class_variable = "shared"
    
    def __init__(self, param):
        self.instance_variable = param
    
    def method(self):
        return self.instance_variable
    
    @staticmethod
    def static_method():
        return "static"
    
    @classmethod
    def class_method(cls):
        return cls.class_variable
```

### 7. Modules and Packages

#### Import System
```python
import module
import module as alias
from module import function
from module import function as alias
from module import *
from package.submodule import function
```

#### Package Structure
```
package/
    __init__.py
    module1.py
    subpackage/
        __init__.py
        module2.py
```

### 8. Advanced Features

#### Decorators
```python
@decorator
def function():
    pass

# Property decorator
class MyClass:
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
```

#### Context Managers
```python
with open('file.txt') as f:
    content = f.read()

# Custom context manager
class MyContext:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # cleanup code
        pass
```

#### Generators and Iterators
```python
def generator():
    yield 1
    yield 2
    yield 3

# Generator expression
gen = (x**2 for x in range(10))

# Iterator protocol
class MyIterator:
    def __iter__(self):
        return self
    
    def __next__(self):
        # return next item or raise StopIteration
        pass
```

#### Async/Await
```python
async def async_function():
    await some_async_operation()
    return result

# Async context manager
async with async_context() as ctx:
    await ctx.do_something()
```

### 9. Comprehensions
```python
# List comprehension
[x**2 for x in range(10) if x % 2 == 0]

# Dict comprehension
{k: v for k, v in items.items() if condition}

# Set comprehension
{x for x in iterable if condition}

# Generator expression
(x for x in iterable if condition)
```

### 10. Built-in Functions (Most Important)

#### Essential Functions
- `len()`, `str()`, `int()`, `float()`, `bool()`
- `print()`, `input()`
- `range()`, `enumerate()`, `zip()`
- `map()`, `filter()`, `reduce()` (from functools)
- `sorted()`, `reversed()`
- `min()`, `max()`, `sum()`
- `any()`, `all()`
- `isinstance()`, `issubclass()`
- `hasattr()`, `getattr()`, `setattr()`, `delattr()`
- `type()`, `id()`, `hash()`

### 11. Standard Library Modules (Key Ones)

#### Core Modules
- `os`: Operating system interface
- `sys`: System-specific parameters
- `json`: JSON encoder/decoder
- `re`: Regular expressions
- `datetime`: Date and time handling
- `collections`: Specialized container datatypes
- `itertools`: Functions for creating iterators
- `functools`: Higher-order functions and tools
- `pathlib`: Object-oriented filesystem paths

#### Data Processing
- `csv`: CSV file reading/writing
- `sqlite3`: SQLite database interface
- `pickle`: Python object serialization

#### Networking and Web
- `urllib`: URL handling modules
- `http`: HTTP modules
- `socket`: Low-level networking interface

### 12. Best Practices for Sub-Agents

#### Code Style
- Follow PEP 8 style guide
- Use meaningful variable names
- Write docstrings for functions and classes
- Use type hints where appropriate

#### Error Handling
- Use specific exception types
- Don't catch all exceptions unless necessary
- Use finally blocks for cleanup

#### Performance Considerations
- Use list comprehensions over loops when appropriate
- Prefer generators for large datasets
- Use `__slots__` for memory-efficient classes
- Consider using `collections.deque` for queues

#### Security Considerations
- Validate input data
- Use `secrets` module for cryptographic randomness
- Avoid `eval()` and `exec()` with untrusted input
- Use `pathlib` for safe file path handling

## Usage Guidelines for Sub-Agents

1. **Leverage Python's Duck Typing**: Focus on object behavior rather than explicit type checking
2. **Use Built-in Functions**: They are optimized and well-tested
3. **Embrace Pythonic Idioms**: Use comprehensions, context managers, and iterators
4. **Handle Exceptions Gracefully**: Use try/except blocks appropriately
5. **Utilize the Standard Library**: It provides robust, tested solutions
6. **Follow the Zen of Python**: Simple is better than complex, explicit is better than implicit

## Common Patterns for AI Agents

### Data Processing Pipeline
```python
def process_data(data):
    return [
        transform(item)
        for item in data
        if validate(item)
    ]
```

### Configuration Management
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    api_key: str
    timeout: int = 30
    debug: bool = False
```

### Async Operations
```python
import asyncio
import aiohttp

async def fetch_data(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

This comprehensive specification should enable sub-agents to fully utilize Python's capabilities across all domains of programming tasks.