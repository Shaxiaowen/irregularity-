"""Main analysis steps for one set of flight records.

`flights` is an iterable of dictionaries with time stamps and the signals of one flight:
    t, current, acceleration, yaw_rate, vx, vy, airflow_reading
`conditions` is a table of operating conditions per flight with a `day` column.
"""
import numpy as np
import pandas as pd

import descriptors as ds
import inference as inf
import learning as ml
import preprocessing as pp

DT, I0 = 0.2, 2.0


def describe_flight(f):
    seg = pp.active_segment(f['current'], I0)
    t = f['t'][seg]
    row, z = {}, {}
    for ch in ('current', 'acceleration', 'yaw_rate'):
        tu, x = pp.resample(t, f[ch][seg], DT)
        z[ch] = pp.detrend_standardize(tu, x)
        row[f'{ch}_dfa'] = ds.dfa_alpha(z[ch])
        row[f'{ch}_sampen'] = ds.sample_entropy(z[ch])
        row[f'{ch}_hf'] = ds.high_frequency_fraction(z[ch], 1 / DT)
        row[f'{ch}_T'] = ds.time_irreversibility(z[ch])
    q, h, h2_se = ds.generalized_hurst(z['current'])
    deg = ds.visibility_degrees(z['current'])
    row.update(current_h2=h[list(q).index(2)], current_h2_se=h2_se, current_vg_max=deg.max(),
               current_vg_tail=ds.degree_tail_slope(deg), current_T_surrogate_z=ds.surrogate_z(z['current'], ds.time_irreversibility))
    _, vx = pp.resample(t, f['vx'][seg], DT); _, vy = pp.resample(t, f['vy'][seg], DT); _, u = pp.resample(t, f['airflow_reading'][seg], DT)
    row['airflow_proxy'], low = pp.airflow_proxy(u, vx, vy)
    row['current_sampen_low_speed'] = ds.sample_entropy(z['current'], valid=low)
    return row


def run(flights, conditions, covariates):
    table = pd.concat([conditions.reset_index(drop=True), pd.DataFrame([describe_flight(f) for f in flights])], axis=1)
    # 1. associations with the airflow proxy, adjusted for operating conditions and clustered by day
    descriptors = [c for c in table if c.startswith(('current_', 'acceleration_', 'yaw_rate_')) and not c.endswith('_se')]
    assoc = pd.DataFrame([(d, *inf.mixed_model(table, d, 'airflow_proxy', covariates)) for d in descriptors],
                         columns=['descriptor', 'beta', 'lo', 'hi', 'p'])
    assoc['q'] = inf.fdr(assoc.p)
    # 2. channel contrast within flights
    contrast = inf.paired_difference(table, 'acceleration_dfa', 'current_dfa', 'airflow_proxy', covariates)
    # 3. propagation of exponent estimation error
    draws = inf.propagate_estimation_error(table, 'current_h2', 'current_h2_se', 'airflow_proxy', covariates)
    # 4. airflow sensing from response descriptors, one collection day held out at a time
    X = table[covariates + descriptors].to_numpy(float); y = table['airflow_proxy'].to_numpy(float); g = table['day'].to_numpy()
    pred = ml.cross_validated_predictions(ml.elastic_net, X, y, g)
    return assoc, contrast, draws, pred
