
def wrap_name(src):
    return f"_orig_{src}"

def add_attr(module, name, target):
    setattr(module, name, target)

def wrap_attr(module, name, wrapper):
    target = getattr(module, name)
    setattr(module, wrap_name(name), target)
    setattr(module, name, wrapper)

def replace_attr(module, name, target):
    wrap_attr(module, name, target)
