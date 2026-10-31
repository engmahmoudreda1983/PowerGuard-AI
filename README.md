# PowerGuard AI: Asset Intelligence Platform

One Streamlit app with one login for all PowerGuard AI modules.

**Main file:** `PowerGuard_AI.app.py` (Python 3.12). Login credentials come from the Streamlit secrets `APP_USERNAME` and `APP_PASSWORD`.

## Pages
- **Command Center**: fleet health, Action / Watch / Normal counts, module tiles, priority alerts, site risk map, latest trips,
  unified asset register, Fleet Status Report (PDF) and CSV downloads.
- **Asset 360**: one site asset seen by every module (for example a transformer: TransformerGuard, GridGuard, PDGuard and
  its RelayGuard relays) with one combined risk = highest module risk + 5 % for every other module above 35 %; Asset 360 Report (PDF).
- **Modules**: TransformerGuard, GridGuard, PDGuard, MotorGuard, RelayGuard, FaultGuard, each with its own pages and downloads.
- **NetworkGuard** (`modules/networkguard`, built in this repository): single-line diagram from per-equipment data (forms or Excel), load flow,
  voltage drop, IEC 60909 short circuit with breaker duty, protection coordination (distance zones, overcurrent grading), N-1 contingency,
  fault study of real faults (from FaultGuard in the same session, from disturbance records and event lists, or by hand), single-line diagram
  from a PDF drawing, equipment data from relay settings files; synced from its own repository (NetworkGuard-AI). Real network data is
  uploaded in the app, never committed.
- **Data source**: demo substation (simulated, anonymised) or the results analyzed on the module pages in the session.

## Structure
- `pg_platform/`: platform code (theme, runner that runs each module as a page with its own session state, fleet results,
  asset register, PDF reports, pages).
- `modules/<module>/`: copies of the module apps, synced from their repositories with `python tools/sync_modules.py` .
- `pg_paths.py`: logo and model paths.
- `PowerGuard_Master.app.py`: the earlier portal (kept unchanged).

Real utility data (records, event lists, settings, spreadsheets) must not be committed to this repository.

© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary.
