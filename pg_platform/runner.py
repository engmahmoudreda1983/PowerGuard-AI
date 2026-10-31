"""Runs a module's own Streamlit script as a page of the platform.

Every module keeps its session state apart: on entering a module, the keys of the module (or platform page) that was
open before are put away and the keys this module had are brought back. Login and platform keys are shared."""
import os
import sys

import streamlit as st

import pg_paths

SHARED = {'logged_in'}
_CODE = {}


def _paths():
    for d in [os.path.join(pg_paths.MODULES, '_common')] + [os.path.join(pg_paths.MODULES, x) for x in sorted(os.listdir(pg_paths.MODULES))]:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def _own(k):
    return not (k in SHARED or str(k).startswith('pg_') or str(k).startswith('_pg'))


def enter(name):
    """Switch the session to the state of `name` (a module key or 'platform')."""
    ss = st.session_state
    store = ss.setdefault('_pg_store', {})
    active = ss.get('_pg_active')
    if active == name:
        return
    if active is not None:
        saved = {k: v for k, v in ss.items() if _own(k)}
        store[active] = saved
        for k in list(saved):
            try:
                del ss[k]
            except Exception:
                pass
    for k, v in store.get(name, {}).items():
        try:
            ss[k] = v
        except Exception:
            pass
    ss['_pg_active'] = name


def module_state(name):
    """The stored session values of a module (read-only view), e.g. its 'register' of analyzed results."""
    ss = st.session_state
    if ss.get('_pg_active') == name:
        return {k: v for k, v in ss.items() if _own(k)}
    return ss.get('_pg_store', {}).get(name, {})


def run(name, script):
    _paths()
    enter(name)
    path = os.path.join(pg_paths.MODULES, name, script)
    code = _CODE.get(path)
    if code is None:
        with open(path, encoding='utf-8') as f:
            code = compile(f.read(), path, 'exec')
        _CODE[path] = code
    g = {'__name__': f'pg_module_{name}', '__file__': path}
    exec(code, g)
