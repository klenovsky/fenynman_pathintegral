# Feynman Path Integral Visualizer

A Streamlit teaching app for visualizing Feynman's path-integral viewpoint with vectorized `numpy` and `scipy` calculations.

## Panels

1. **Free particle**
   - sampled paths between fixed endpoints
   - action distribution
   - complex phase interference

2. **Harmonic oscillator**
   - endpoint-conditioned classical path
   - sampled paths in a confining potential
   - interference around the stationary path

3. **Two path families**
   - double-slit-style intuition
   - two bundles of paths contributing separate complex amplitudes

4. **Real time vs imaginary time**
   - oscillatory real-time factors `exp(iS/ħ)`
   - exponentially damped Euclidean weights `exp(-S_E/ħ)`

## Numerical approach

The app is intentionally didactic rather than mathematically exact in the continuum sense.

- time is discretized on a fixed grid,
- candidate paths are generated in batches around reference paths,
- `scipy.interpolate.CubicSpline` is used in vectorized form,
- actions are evaluated on whole path ensembles with `numpy`,
- no Python loops are used over the full path batch during action evaluation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload this folder to GitHub and deploy `app.py` on Streamlit Community Cloud.
