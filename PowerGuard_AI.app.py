"""PowerGuard AI: Asset Intelligence Platform.

One login, one Command Center, Asset 360 and the seven modules (TransformerGuard, GridGuard, PDGuard, MotorGuard,
RelayGuard, FaultGuard, NetworkGuard) as pages. Login credentials come from the Streamlit secrets APP_USERNAME and APP_PASSWORD.
© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary."""
import hmac
import warnings

import streamlit as st

warnings.filterwarnings('ignore')
st.set_page_config(page_title='PowerGuard AI', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')

from pg_platform import fleet as FL  # noqa: E402
from pg_platform import pages as P  # noqa: E402
from pg_platform import runner  # noqa: E402
from pg_platform import theme as T  # noqa: E402

T.inject()
ss = st.session_state
ss.setdefault('logged_in', False)


def _secrets_ok():
    try:
        return 'APP_USERNAME' in st.secrets and 'APP_PASSWORD' in st.secrets
    except Exception:
        return False


def login():
    st.markdown("<style>section[data-testid='stSidebar']{display:none} [data-testid='stHeader']{background:transparent}"
                "[data-testid='stForm']{background:#fff;border-radius:0 0 22px 22px;border:none;margin-top:-22px;padding:6px 30px 24px 30px;max-width:430px;margin-left:auto;margin-right:auto}"
                ".pg-login{border-radius:22px 22px 0 0 !important;box-shadow:none !important;border:none !important}"
                ".stApp{background:radial-gradient(900px 500px at 15% 0%, #1d5fb3 0%, transparent 60%),"
                "radial-gradient(900px 600px at 100% 100%, #2f80ed55 0%, transparent 60%), linear-gradient(135deg,#0b2e59,#081b36)}</style>",
                unsafe_allow_html=True)
    lg = T.logo_b64()
    _, c, _ = st.columns([1, 1.15, 1])
    with c:
        st.markdown(f"<div class='pg-login'><img src='data:image/png;base64,{lg}'><h2>PowerGuard AI</h2>"
                    "<p>Asset Intelligence Platform</p></div>", unsafe_allow_html=True)
        with st.form('pg_login'):
            u = st.text_input('Username', placeholder='Username', label_visibility='collapsed')
            p = st.text_input('Password', type='password', placeholder='Password', label_visibility='collapsed')
            if st.form_submit_button('Sign in', type='primary', width='stretch'):
                if not _secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(u, st.secrets['APP_USERNAME']) and hmac.compare_digest(p, st.secrets['APP_PASSWORD']):
                    ss['logged_in'] = True
                    st.rerun()
                else:
                    st.error('Invalid username or password.')
        st.markdown("<div style='text-align:center;color:#9fb7da;font-size:12px;margin-top:10px'>"
                    "TransformerGuard · GridGuard · PDGuard · MotorGuard · RelayGuard · FaultGuard · NetworkGuard<br><br>"
                    "© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary.</div>", unsafe_allow_html=True)


if not ss['logged_in']:
    login()
    st.stop()

SCRIPTS = {'transformerguard': 'AI_Transformer_Health_V2.app.py', 'gridguard': 'GridGuard_AI.app.py', 'pdguard': 'PDGuard_AI.app.py',
           'motorguard': 'MotorGuard_AI.app.py', 'relayguard': 'RelayGuard_AI.app.py', 'faultguard': 'FaultGuard_AI.app.py',
           'networkguard': 'NetworkGuard_AI.app.py'}
PAGES = {}


def _module_page(key):
    def page():
        runner.run(key, SCRIPTS[key])
    page.__name__ = f'pg_{key}'
    return page


def _platform_page(fn, name):
    def page():
        fn(PAGES)
    page.__name__ = name
    return page


PAGES['home'] = st.Page(_platform_page(P.command_center, 'command_center'), title='Command Center', icon=':material/space_dashboard:',
                        url_path='command-center', default=True)
PAGES['asset360'] = st.Page(_platform_page(P.asset360, 'asset_360'), title='Asset 360', icon=':material/travel_explore:', url_path='asset-360')
for k, (nm, ic, _) in FL.MODULE_INFO.items():
    PAGES[k] = st.Page(_module_page(k), title=nm, icon=ic, url_path=k)
PAGES['about'] = st.Page(_platform_page(P.about, 'about'), title='About', icon=':material/info:', url_path='about')

nav = st.navigation({'Overview': [PAGES['home'], PAGES['asset360']],
                     'Asset health': [PAGES[k] for k in ('transformerguard', 'gridguard', 'pdguard', 'motorguard')],
                     'Protection and faults': [PAGES['relayguard'], PAGES['faultguard']],
                     'Network studies': [PAGES['networkguard']],
                     'Platform': [PAGES['about']]})

try:
    st.logo(T.small_logo_path(), size='large')
except Exception:
    pass
with st.sidebar:
    st.markdown("<div style='margin:-6px 4px 10px 4px'><div style='font-weight:800;font-size:1.05rem;color:#fff'>PowerGuard AI</div>"
                "<div style='font-size:.78rem;color:#8fb4e8'>Asset Intelligence Platform</div></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    user = st.secrets.get('APP_USERNAME', 'user') if _secrets_ok() else 'user'
    st.markdown(f"<div style='border-top:1px solid rgba(255,255,255,.15);padding-top:10px;font-size:.82rem'>👤 Signed in as <b>{user}</b></div>",
                unsafe_allow_html=True)
    if st.button('Sign out', icon=':material/logout:', width='stretch', key='pg_logout'):
        ss['logged_in'] = False
        ss.pop('_pg_store', None)
        st.rerun()
    st.markdown("<div style='font-size:.7rem;color:#7f9cc6;margin-top:8px'>v1.0 · © 2026 PowerGuard AI</div>", unsafe_allow_html=True)


# sign out at the top of every page (the sidebar can be closed, on phones it is hidden)
_l, _r = st.columns([6, 1.3])
with _r:
    with st.popover(f'👤 {user}', width='stretch'):
        st.caption('PowerGuard AI · Asset Intelligence Platform')
        if st.button('Sign out', icon=':material/logout:', width='stretch', type='primary', key='pg_logout_top'):
            ss['logged_in'] = False
            ss.pop('_pg_store', None)
            st.rerun()

nav.run()
