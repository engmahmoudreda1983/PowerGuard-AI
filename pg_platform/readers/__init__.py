"""Readers for relay settings files (read-only).

micom_set  - MiCOM S1 Agile / S1 Studio Courier settings (.set): line data, CT/VT, zones.
px30_x3v   - MiCOM Px30 settings (.x3v): transformer rating, vector group, differential settings.

The readers never write to a settings file. Real network data stays on the user's
computer (PowerGuard_Data); only this code lives in the repository.
"""
