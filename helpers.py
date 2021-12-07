import functools
from datetime import datetime


def pprinter(program_name: str):
    """
    Binds @program_name to print statement for cleaner print statements
    """
    def printer(text):
        print(
            f'[{program_name:19} | {datetime.utcnow().strftime("%I:%M:%S.%f")}]: {text}'
        )

    return printer


# TODO: is *args necessary?
def rgetattr(obj, attr, *args):
    """
    Takes a string attr1.attr2.(...).attrN and returns the bottom attribute
    source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
