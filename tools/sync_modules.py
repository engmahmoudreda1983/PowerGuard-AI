"""Copy the PowerGuard AI modules from their own repositories into this platform and adapt them to run as pages.

    python tools/sync_modules.py [module ...]   (expects the module repositories next to this one)

Each module keeps its own code; the copy only
  * drops st.set_page_config (the platform sets it once),
  * points logo and model files to the platform folders (pg_paths),
  * renames report_pdf.py to <module>_report_pdf.py so the modules do not shadow each other,
  * shares one comtrade_features.py,
  * hides the module's own Logout button (the platform has one sign-out).
Run it again after a module changes, then commit."""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(ROOT)
MODULES = {
    'transformerguard': ('powerguard-ai-transformer-health-life-predictort', 'AI_Transformer_Health_V2.app.py',
                         ['powerguard_core.py', 'rf_health_model.pkl', 'rf_thermal_model.pkl']),
    'gridguard': ('gridguard_ai', 'GridGuard_AI.app.py', ['gridguard_core.py', 'train_gridguard_model.py', 'report_pdf.py', 'gridguard_model.pkl']),
    'pdguard': ('pdguard-ai', 'PDGuard_AI.app.py', ['pdguard_core.py', 'pd_classifier.pkl']),
    'motorguard': ('motorguard_ai', 'MotorGuard_AI.app.py', ['motorguard_core.py', 'thermal_model.pkl']),
    'relayguard': ('relayguard_ai', 'RelayGuard_AI.app.py', ['relayguard_core.py', 'omicron_report.py', 'report_pdf.py', 'record_anomaly_model.pkl']),
    'faultguard': ('faultguard_ai', 'FaultGuard_AI.app.py', ['faultguard_core.py', 'report_pdf.py', 'faultguard_model.pkl']),
    'networkguard': ('networkguard_ai', 'NetworkGuard_AI.app.py', ['networkguard_core.py', 'networkguard_plots.py', 'networkguard_import.py',
                                                                   'iec60909_4_example.py', 'report_pdf.py']),   # FaultGuard engine comes from modules/faultguard
}
SHARED = [('relayguard_ai', 'comtrade_features.py')]


def patch(text, key):
    text = re.sub(r'^st\.set_page_config\(.*\)\s*$', 'pass  # page config is set once by the PowerGuard AI platform', text, flags=re.M)
    text = text.replace("Image.open('logo.png')", 'Image.open(pg_paths.LOGO)')
    text = re.sub(r"(joblib\.load|open|load_model|load_record_model)\('([\w]+\.pkl)'", r"\1(pg_paths.model('\2')", text)
    text = re.sub(r'^import report_pdf as', f'import {key}_report_pdf as', text, flags=re.M)
    text = text.replace('if st.button("Logout 🚪"):', 'if False:  # sign out is in the platform header and sidebar')
    text = text.replace('<h1>⚡ Transformer & Reactor Guard AI</h1>', '<h1>⚡ TransformerGuard AI: Transformer and Reactor Health</h1>')
    if 'pg_paths.' in text and 'import pg_paths' not in text:
        text = re.sub(r'^import streamlit as st\s*$', 'import streamlit as st\nimport pg_paths', text, count=1, flags=re.M)
        if 'import pg_paths' not in text:
            text = 'import pg_paths\n' + text
    return text


def main(only=None):
    dst_root = os.path.join(ROOT, 'modules')
    for key, (repo, app, files) in MODULES.items():
        if only and key not in only:
            continue
        src = os.path.join(SRC, repo)
        if not os.path.isdir(src):
            sys.exit(f'missing repository folder: {src}')
        dst = os.path.join(dst_root, key)
        os.makedirs(dst, exist_ok=True)
        for f in [app] + files:
            s, d = os.path.join(src, f), os.path.join(dst, f)
            if f == 'report_pdf.py':
                d = os.path.join(dst, f'{key}_report_pdf.py')
            if f.endswith('.py'):
                with open(s, encoding='utf-8') as fh:
                    t = patch(fh.read(), key)
                with open(d, 'w', encoding='utf-8') as fh:
                    fh.write(t)
            else:
                shutil.copy2(s, d)
        print(f'{key}: {len(files) + 1} files from {repo}')
    common = os.path.join(dst_root, '_common')
    os.makedirs(common, exist_ok=True)
    for repo, f in SHARED:
        shutil.copy2(os.path.join(SRC, repo, f), os.path.join(common, f))
    print('shared: comtrade_features.py')


if __name__ == '__main__':
    main(sys.argv[1:] or None)   # e.g. python tools/sync_modules.py networkguard
