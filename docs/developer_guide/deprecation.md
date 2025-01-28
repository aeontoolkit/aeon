# Deprecation Policy

`aeon` [releases](https://github.com/aeon-toolkit/aeon/releases) follow [semantic versioning](https://semver.org). A release number
denotes `<major>.<minor>.<patch>` versions.

Broadly, if a change could unexpectedly cause code using `aeon` to crash when updating
to the next version, then it should be deprecated to give the user a chance to prepare.

When to deprecate:
- Removal or renaming of public classes or functions
- Removal or renaming of public class parameters or function arguments
- Addition of positional arguments without default values

Deprecation warnings should be included for at least one full minor version cycle before
change or removal. If an item is deprecated on the release of `v0.6.0`, it can be
removed  in `v0.7.0`. If an item is deprecated between `v0.6.0` and `v0.7.0`
(i.e. `v0.6.1`), it can be removed in `v0.8.0`.

Note that the deprecation policy does not necessarily apply to modules we class as still
experimental. Currently experimental modules are:

- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.imbalance`

When we introduce a new module, we may classify it as experimental until the API is
stable. We will try to not make drastic changes to experimental modules, but we need
to retain the freedom to be more agile with the design in these cases.

## Deprecation Process

To deprecate functions and classes, write a "TODO" comment stating the version the code
should be removed in and raise a warning using the [deprecated package](https://deprecated.readthedocs.io/en/latest/index.html). This
raises a `FutureWarning` saying that the functionality has been deprecated. Import
from `deprecated.sphinx` so the  deprecation message is automatically added to the
documentation.

When renaming items, the functionality should ideally already be available with the new
name when the deprecation warning is added. For example, including both the old and new
name for a positional argument, or both functions/classes with the old and new names.
This is not always possible, but it is good practice to do so.

In most cases not necessary to use the `deprecated` package when renaming or removing
function and class keyword arguments. The default value of the argument can be set to
`"deprecated"`. If this value is changed, a `FutureWarning` can be raised. This
isolates the deprecation warning to the argument, rather than the whole function or
class. If renaming, the new keyword argument can be added alongside this, with the
warning directing users to use the new keyword argument.

If the next version number has not been decided, use the next minor version number
for the deprecated package `version` parameter.

## Examples

### Functions and Methods

Deprecate a function.

```python
from deprecated.sphinx import deprecated

# TODO: remove in v0.7.0
@deprecated(
    version="0.6.0",
    reason="my_function will be removed in v0.7.0.",
    category=FutureWarning,
)
def my_function(x, y):
    """Return x + y."""
    return x + y
```

Deprecate a function to add a new positional argument. In certain cases, you can add a
keyword argument with a default value, to ease the transition.

```python
from deprecated.sphinx import deprecated

# TODO: remove in v0.7.0 and remove z default
@deprecated(
    version="0.6.0",
    reason="my_function will be include a third positional argument 'z' in v0.7.0, used for reasons.",
    category=FutureWarning,
)
def my_function(x, y, z=0):
    """Return x + y + z."""
    return x + y + z
```

Deprecate a class method to remove a positional argument.

```python
from deprecated.sphinx import deprecated

class MyClass:
    """Contains my_method."""

    # TODO: remove in v0.7.0
    @deprecated(
        version="0.6.0",
        reason="my_method first positional argument 'x' will be removed in v0.7.0.",
        category=FutureWarning,
    )
    def my_method(self, x, y):
        """Return y."""
        return y
```

### Classes

Deprecate a class.

Since this example is deprecated on a patch release, it cannot be removed from the next
minor release.

```python
from deprecated.sphinx import deprecated

# TODO: remove in v0.8.0
@deprecated(
    version="0.6.1",
    reason="MyClass will be removed in v0.8.0.",
    category=FutureWarning,
)
class MyClass:
    """Exists (for now)."""
    pass
```

Deprecate a public class attribute. If we are renaming, we could add the new name and
direct users to use that instead while updating both.

```python
from deprecated.sphinx import deprecated

# TODO: remove in v0.7.0
@deprecated(
    version="0.6.0",
    reason="The my_attribute attribute will be removed in v0.7.0.",
    category=FutureWarning,
)
class MyClass:
    """Contains my_attribute."""

    my_attribute = 1
```

Deprecate a class parameter.

```python
import warnings

class MyClass:
    """Contains x and y.

    Parameters
    ----------
    x : int, default=1
        The x parameter.
    y : int, default="deprecated"
        The y parameter.

        Deprecated and will be removed in v0.7.0.
    """

    # TODO remove 'y' in v0.7.0
    def __init__(self, x=1, y="deprecated"):
        self.x = x
        if y != "deprecated":
            warnings.warn("The 'y' parameter will be removed in v0.7.0.", FutureWarning)
```

### Renaming

Deprecate a class parameter to rename it.

```python
import warnings

class MyClass:
    """Does stuff.

    Parameters
    ----------
    x : int, default=1
        The x parameter.
    new_param : int, default=2
        New identifier for old_param.
    old_param : int, default="deprecated"
        Old identifier for new_param. If used will set new_param.

        Deprecated and will be removed in v0.7.0.
    """

    # TODO remove 'old_param' in v0.7.0
    def __init__(self, x=1, new_param=2, old_param="deprecated"):
        self.x = x
        self.new_param = new_param
        if old_param != "deprecated":
            warnings.warn("The 'old_param' parameter will be removed in v0.7.0. Use 'new_param' instead.", FutureWarning)
            self.new_param = old_param
```

Deprecate a class to rename it.

```python
from deprecated.sphinx import deprecated

class MyNewClass:
    """Exists."""

    @staticmethod
    def return_one():
        """Return 1."""
        return 1


# TODO: remove in v0.7.0
@deprecated(
    version="0.6.0",
    reason="The MyOldClass class has been renamed to MyNewClass and will be removed in v0.7.0.",
    category=FutureWarning,
)
class MyOldClass(MyNewClass):
    """Exists."""
    pass
```

Deprecate a function to rename it.

```python
from deprecated.sphinx import deprecated

def my_new_function(x, y):
    """Return x + y."""
    return x + y


# TODO: remove in v0.7.0
@deprecated(
    version="0.6.0",
    reason="my_old_function has been renamed to my_new_function and will be removed in v0.7.0.",
    category=FutureWarning,
)
def my_old_function(x, y):
    """Return x + y."""
    return my_new_function(x, y)
```
