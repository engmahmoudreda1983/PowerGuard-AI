import pg_paths
"""Train the GridGuard AI load model and measure its accuracy on data it never saw.

    python train_gridguard_model.py                     # demo substation (synthetic)
    python train_gridguard_model.py register.csv loads.csv weather.csv [split-date]

The app can also re-train the model on uploaded client files (Model & Accuracy page); this script does the same."""
import sys, pickle
import numpy as np, pandas as pd
import gridguard_core as g


def evaluate(bundle, reg, loads, weather, start, end, step_days=3, seed=0):
    """Forecast from many origins in [start, end] with a realistic weather forecast and compare with the actual load."""
    rows = []
    W = weather.assign(_t=pd.to_datetime(weather['Timestamp']))
    for r in reg.to_dict('records'):
        L = loads[loads['Transformer ID'].astype(str) == str(r['Transformer ID'])]
        s = L.assign(_t=pd.to_datetime(L['Timestamp'])).set_index('_t')['Load (MVA)'].astype(float).sort_index()
        for origin in pd.date_range(start, end, freq=f'{step_days}D'):
            if origin + pd.Timedelta(days=g.HORIZON_DAYS) > s.index[-1] or origin - pd.Timedelta(days=8) < s.index[0]:
                continue
            past = s[s.index < origin]
            wf = g.weather_forecast_from(weather, origin, seed=seed)
            Wp = W[W['_t'] < origin]
            fc = g.forecast(bundle, r, past.index, past.to_numpy(), Wp['_t'], Wp['Ambient (C)'], wf['Timestamp'], wf['Ambient (C)'], origin)
            act = s.reindex(fc['ts']).to_numpy()
            for i in range(len(act)):
                rows.append((r['Transformer ID'], origin, i // 24 + 1, fc['ts'][i], act[i], fc['load'][i]))
    E = pd.DataFrame(rows, columns=['Transformer ID', 'Origin', 'Day Ahead', 'Timestamp', 'Actual', 'Forecast'])
    E['APE'] = (E['Forecast'] - E['Actual']).abs() / E['Actual'] * 100
    E['Ratio Error'] = E['Forecast'] / E['Actual'] - 1
    return E


def fit(reg, loads, weather, split, seed=0):
    tab = g.training_table(loads, reg, weather)
    train = tab[tab.index < pd.Timestamp(split)]
    m = g.train_model(train, seed=seed)
    bundle = {'model': m, 'features': g.FEATURES, 'version': 1, 'trained_until': str(pd.Timestamp(split).date()),
              'n_rows': int(len(train)), 'n_transformers': int(train['Transformer ID'].nunique()), 'err_sd_by_day': [0.04] * g.HORIZON_DAYS}
    return bundle


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        reg, loads, weather = (pd.read_csv(f) for f in sys.argv[1:4])
        split = sys.argv[4] if len(sys.argv) > 4 else str((pd.to_datetime(loads['Timestamp']).max() - pd.Timedelta(days=120)).date())
        end = pd.to_datetime(loads['Timestamp']).max() - pd.Timedelta(days=g.HORIZON_DAYS)
    else:
        d = g.demo_data(); reg, loads, weather = d['register'], d['loads'], d['weather']
        split, end = '2025-05-01', pd.Timestamp('2026-08-24')
    bundle = fit(reg, loads, weather, split)
    E = evaluate(bundle, reg, loads, weather, split, end)
    sd = E.groupby('Day Ahead')['Ratio Error'].std().reindex(range(1, g.HORIZON_DAYS + 1)).ffill().fillna(0.05)
    bundle['err_sd_by_day'] = [round(float(v), 4) for v in sd]
    bundle['accuracy'] = {'test_from': str(pd.Timestamp(split).date()), 'test_to': str(pd.Timestamp(end).date()),
                          'mape_by_day': [round(float(v), 2) for v in E.groupby('Day Ahead')['APE'].mean()],
                          'mape_by_tr': {k: round(float(v), 2) for k, v in E[E['Day Ahead'] == 1].groupby('Transformer ID')['APE'].mean().items()},
                          'peak_err_day1': round(float(E[E['Day Ahead'] == 1].groupby(['Transformer ID', E['Timestamp'].dt.normalize()])
                                                       .apply(lambda x: abs(x['Forecast'].max() / x['Actual'].max() - 1) * 100).mean()), 2),
                          'n_forecasts': int(E.groupby(['Transformer ID', 'Origin']).ngroups)}
    with open(pg_paths.model('gridguard_model.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    print('trained on', bundle['n_rows'], 'rows until', split)
    print(bundle['accuracy']); print('error sd by day', bundle['err_sd_by_day'])
