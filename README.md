# Multirotor responses to stochastic aerodynamic excitation

Code for the main analysis steps of the manuscript *"Scaling, irregularity and time irreversibility in multirotor responses to stochastic aerodynamic excitation: multiscale signal analysis and interpretable learning from operational flights"*.

## Main steps

1. **Preprocessing** (`preprocessing.py`): active flight segment, uniform resampling, polynomial detrending and standardization; airflow proxy from anemometer readings at low horizontal groundspeed.
2. **Descriptors** (`descriptors.py`): DFA and MFDFA scaling exponents with standard errors, sample entropy (optionally restricted to selected periods), correlation dimension, visibility-graph degree statistics, time irreversibility, high-frequency power fraction, spectral exponent, 0–1 test, Lyapunov exponent, IAAFT surrogates, transfer entropy, coherence and ARX residuals.
3. **Inference** (`inference.py`): random-intercept mixed-effects models with standardized coefficients, within/between decomposition, leave-one-group-out refits, false discovery rate correction, paired channel contrasts and Monte Carlo propagation of exponent estimation error.
4. **Learning** (`learning.py`): elastic net, gradient-boosted trees and a one-dimensional convolutional network under leave-one-group-out validation, group-block bootstrap comparison and SHAP attribution.
5. **Example** (`example_pipeline.py`): how the steps are combined for a set of flights.

## Requirements

Python 3.12 and the packages in `requirements.txt`.

## License

MIT License (see `LICENSE`).
