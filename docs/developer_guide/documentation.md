# Developing Documentation

`aeon`'s documentation standards include:

- Documenting code using `numpydoc` docstring conventions
- Adding new public functionality to the [api_reference](https://www.aeon-toolkit.org/en/stable/api_reference.html).

More detailed information on `aeon`'s documentation format is provided below.

## Docstring conventions

`aeon` uses the `numpydoc` Sphinx extension and follows [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

To ensure docstrings meet expectations, `aeon` uses a combination of validations built
into `numpydoc` and `pydocstyle` `pre-commit` checks (set to the NumPy convention) and
automated testing of docstring examples to ensure the code runs without error.

Beyond basic NumPy docstring formatting conventions, developers should aim to:

- Ensure all parameters (classes, functions, methods) and attributes (classes) are
documented completely and consistently
- Add a `See Also` section that references related `aeon` code as applicable
- Include citations to relevant sources in a `References` section
- Include an `Examples` section that demonstrates at least basic functionality all
public code
- The docstrings are rendered into `.rst` files and should be written taking this into
account. For example, two ` characters are required for a code block instead of the
one used in Markdown.

In many cases, a parameter, attribute return object, or error may be described in many
docstrings across aeon. To avoid confusion, developers should try to make sure their
docstrings are as consistent as possible to existing docstring descriptions.

`aeon` should generally include `numpydoc` section in the following order as applicable:

1. Summary
2. Extended Summary
3. Parameters
4. Attributes (classes only)
5. Returns/Yields (functions/methods only)
6. Raises (functions/methods only)
7. See Also
8. Notes
9. References
10. Examples

### Summary and Extended Summary

The summary should be a single line, followed by a extended summary. The extended
summary should include a user-friendly explanation of the code functionality,
i.e. a short, user-friendly synopsis of the algorithm being implemented or a
high-level summary of the estimator components.

### Parameters and Attributes

All parameters and fitted attributes (any public attribute i.e. those set in fit and
ending with `_`) should be listed in the docstring. Each parameter and attribute
should include a description, type, and default value (if applicable). For example:

```clean
n_jobs : int, default=1
    The number of jobs to run in parallel for both ``fit`` and ``predict``.
    ``-1`` means using all processors.
```

Parameters without default values or attributes do not need to include a default value:

```clean
n_cases_ : int
    Number of train instances in data passed to ``fit``.
```

### See Also

This section should reference other `aeon` code related to the code being documented.
For example, the Catch22 pipeline classifier may reference the Catch22 feature
transformation and the pipeline regressor.

```clean
ContractableBOSS
    Variant of the BOSS classifier.
WEASEL
    SFA based pipeline extending from BOSS.
SFA
    The Symbolic Fourier Approximation feature transformation used in BOSS.
```

### Notes

The notes section can information which is useful but does not fit into the other
sections or extended summary. At the discretion of developers. Some examples are:

- Links to alternative implementations of the code that are external to `aeon`
- Links to code used or taken inspiration from (sometimes this is better in the
extended summary)
- Explanations of quirks or limitations of the code

### References

`aeon` estimators that implement a published algorithm should generally include
citations to the original article (including arxiv etc.). Other papers relevant to the
code such as evaluations or extensions can also be included.

References must be included in the following format:

```clean
.. [1] Some research article, link or other type of citation.
   Long references wrap onto multiple lines, but you need to
   indent them so they start aligned with opening bracket on first line.
```

The `.. [*]` must be included at the start of the reference for it to render correctly.
include whitespace for other lines as shown. The reference label should be incremented
by 1 for each new reference.

To link to the reference labelled as `[1]`, you use `[1]_` elsewhere in the docstring.
For multiple contiguous references follow the format `[1]_, [2]_`. This only works
within the same docstring. Include whitespace between the reference label and other
text in the docstring.

### Examples

Most public code in `aeon` should include an examples section. At a minimum, this should
include a single example that illustrates basic functionality. The examples should use
either a built-in `aeon` dataset or other simple data (e.g. randomly generated data)
where possible. Examples should also be designed to run quickly where possible.

```python
>>> import numpy as np
>>> from aeon.distances import dtw_distance
>>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
>>> dtw_distance(x, y) # 1D series
768.0

>>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
>>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
>>> dtw_distance(x, y) # 2D series with 3 channels, unequal length
564.0
```

`>>>` is used to indicate a line of code. Lines can be continued with `...`, for
example, if you have a long import you may want to do:

```python
>>> from aeon.classification.dictionary_based import (
...     BOSSEnsemble
... )
```

## Examples of good `aeon` docstrings

Here are a few examples of `aeon` code with good documentation.

### Estimators

[BOSSEnsemble](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.dictionary_based.BOSSEnsemble.html#aeon.classification.dictionary_based.BOSSEnsemble)

### Functions

[dtw_distance](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.dtw_distance.html)

## Documentation build

We use [sphinx](https://www.sphinx-doc.org/) to build our documentation and
[readthedocs](https://readthedocs.org/projects/aeon-toolkit/) to host it. You can find
our latest documentation [here](https://www.aeon-toolkit.org/en/latest/).

The source files can be found in [`docs/`](https://github.com/aeon-toolkit/aeon/tree/main/docs/).
The main configuration file for sphinx is [`conf.py`](https://github.com/aeon-toolkit/aeon/blob/main/docs/conf.py)
and the main page is [`index.md`](https://github.com/aeon-toolkit/aeon/blob/main/docs/index.md).
To add new pages, you need to add a new `.md` (or `.rst`, but preferably Markdown)
file and include it in a `toctree`  to include it in the sidebar.

To build the documentation locally, you need to install a few extra dependencies
listed in [pyproject.toml](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml).

1. To install documentation dependencies from the root directory, run:

```powershell
pip install --editable .[docs]
```

2. Swap to the documentation directory:

```powershell
cd docs
```

> **Note:** If you are using Linux or MacOS, you'll have to install [**Pandoc**](https://pandoc.org/) as a dependency. Install it using the following command:
>
> If you're using any Debian based system, you can use the following command to install Pandoc:
> ```bash
> sudo apt install pandoc
> ```
>
> If you're using MacOS, you can use the following command to install Pandoc:
> ```bash
> brew install pandoc
> ```
>
> For all other operating systems, please checkout the [Pandoc installation guide](https://pandoc.org/installing.html).
>

3. To build the website locally, run:

```powershell
make html
```
For Windows, instead use:

```powershell
make.bat html
```
This will generate HTML documentation in `docs/_build/html`. Repeat step 3 to
regenerate the files if you make any changes.
