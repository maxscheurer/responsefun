# taken from respondo

import numpy as np


def select_property_method(matrix):
    if matrix.method.level < 3:
        return matrix.method
    else:
        # Auto-select second-order properties for third-order calc
        return matrix.method.at_level(2)


# taken from adcc (written by mfherbst)
def expand_test_templates(arguments, template_prefix="template_"):
    """
    Expand the test templates of the class cls using the arguments
    provided as a list of tuples to this function
    """
    parsed_args = []
    for args in arguments:
        if isinstance(args, tuple):
            parsed_args.append(args)
        else:
            parsed_args.append((args, ))

    def inner_decorator(cls):
        for fctn in dir(cls):
            if not fctn.startswith(template_prefix):
                continue
            basename = fctn[len(template_prefix):]
            for args in parsed_args:
                newname = "test_" + basename + "_"
                newname += "_".join(str(a) for a in args)

                # Call the actual function by capturing the
                # fctn and args arguments by-value using the
                # trick of supplying them as default arguments
                # (which are evaluated at definition-time)
                def caller(self, fctn=fctn, args=args):
                    return getattr(self, fctn)(*args)
                setattr(cls, newname, caller)
        return cls
    return inner_decorator


# taken from adcc (written by mfherbst)
def assert_allclose_signfix(actual, desired, atol=0, **kwargs):
    """
    Call assert_allclose, but beforehand normalise the sign
    of the involved arrays (i.e. the two arrays may differ
    up to a sign factor of -1)
    """
    actual, desired = normalise_sign(actual, desired, atol=atol)
    np.testing.assert_allclose(actual, desired, atol=atol, **kwargs)


# taken from adcc (written by mfherbst)
def normalise_sign(*items, atol=0):
    """
    Normalise the sign of a list of numpy arrays
    """
    def sign(item):
        flat = np.ravel(item)
        flat = flat[np.abs(flat) > atol]
        if flat.size == 0:
            return 1
        else:
            return np.sign(flat[0])
    desired_sign = sign(items[0])
    return tuple(desired_sign / sign(item) * item for item in items)
