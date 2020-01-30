import os
import sys
import shutil
import itertools


def aprint(*args, func=print, fmt='d', fname=None, **kwargs):
    """
    Advanced printing with certain colors and font styles. Printing to files
    is also possible.
    :param args: "args to pass to param func"
    :param func: functional, optional, "function that prints stuff"
    :param fmt: str, optional, one out of fmts, "format indicator"
    :param fname: str, optional, "filename to print to", if set, then param
        fmt is ignored
    :param kwargs: "kwargs to pass to param func"
    """

    fmts = {'h': '\x1b[42m\x1b[90m',  # heading
            'bh': '\x1b[1m\x1b[42m\x1b[90m',  # bold heading
            'i': '\x1b[32m',  # info
            'bi': '\x1b[1m\x1b[32m',  # bold info
            'w': '\x1b[93m',  # warning
            'bw': '\x1b[1m\x1b[93m',  # bold warning
            'e': '\x1b[31m',  # error
            'be': '\x1b[1m\x1b[31m',  # bold error
            'b': '\x1b[1m',  # bold
            'd': '\x1b[0m'}  # default

    assert(callable(func))
    assert(fmt in fmts.keys())
    assert(isinstance(fname, str) or fname is None)

    if fname is None:
        # print to console in specified format
        sys.stdout.write(fmts[fmt])
        func(*args, **kwargs)
        sys.stdout.write(fmts['d'])
    else:
        # print to file
        with open(fname, 'a+') as sys.stdout:
            func(*args, **kwargs)
        sys.stdout = sys.__stdout__


class Path:
    def __init__(self, path, replace=False):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if replace:
                assert(os.path.isdir(path))
                shutil.rmtree(path)
                os.makedirs(path)

    def join(self, *args, rename_if_exists=False):
        file = os.path.join(self.path, *args)
        if rename_if_exists:
            k = itertools.count(start=1)
            while os.path.exists(file):
                fname, ext = os.path.splitext(file)
                file = '{}_{}{}'.format(fname, next(k), ext)
        return file

    def __str__(self):
        return self.path
