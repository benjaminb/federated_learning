import functools


# TODO: is *args necessary?
def rgetattr(obj, attr, *args):
    """
    Takes a string attr1.attr2.(...).attrN and returns the bottom attribute
    source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
