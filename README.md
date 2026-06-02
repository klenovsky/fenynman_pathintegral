# Feynman Path Integral Explorer

A bilingual Streamlit teaching app for visualizing Feynman's path-integral picture of quantum mechanics.

## Main features

- English / Czech interface
- Free-particle panel
- Harmonic-oscillator panel
- Double-slit-inspired two-family interference panel
- Real-time vs imaginary-time comparison
- Reset buttons per section
- Plotly Play animations showing how individual path contributions build a cumulative complex amplitude
- Vectorized `numpy` / `scipy` implementation suitable for Streamlit Community Cloud

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

1. Upload this folder to a GitHub repository.
2. Create a new app in Streamlit Community Cloud.
3. Choose `app.py` as the main file.
4. Deploy.

## Notes

This is a didactic visualization tool, not a rigorous continuum path-integral solver. The app samples large ensembles of smooth trial paths, evaluates discrete actions in a vectorized way, and visualizes how complex amplitudes interfere.
