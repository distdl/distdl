import distdl.functional.interpolate._cpp as interp_module


def interp(x, y):

    interp_module.forward(x, y)
