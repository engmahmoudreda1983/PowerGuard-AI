"""File locations of the PowerGuard AI platform, used by the module pages."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(ROOT, 'assets', 'logo.png')
MODULES = os.path.join(ROOT, 'modules')


def model(name):
    """Full path of a module's model file (searched in every module folder)."""
    for d in sorted(os.listdir(MODULES)):
        p = os.path.join(MODULES, d, name)
        if os.path.exists(p):
            return p
    return name
