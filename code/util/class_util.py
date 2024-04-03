
import inspect
from functools import wraps

def member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__: 
    :returns: 
    :rtype: 

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def member_wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)
    return member_wrapper

def pt_member_initialize(wrapped__init__):
    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def member_wrapper(self, *args, **kargs):
        super(self.__class__, self).__init__()
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])
        wrapped__init__(self, *args, **kargs)
    return member_wrapper

