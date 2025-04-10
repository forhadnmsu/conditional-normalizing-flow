# Conditional Normalizing Flow for Simulation-to-Data Morphing

## Overview
This project implements a **Conditional Normalizing Flow (CNF)** to transform simulated Monte Carlo (MC) data into distributions that match experimental data. The model uses **affine coupling** and **permutation transforms** to learn the mapping between the source (MC) and target (experimental) distributions. After training, the flow can morph MC data into distributions resembling experimental measurements, providing improved accuracy for simulations in fields such as particle physics.

## Key Features
- **Conditional Normalizing Flows**: Leverages the power of normalizing flows with conditional information to model complex distributions.
- **Affine Coupling Layers**: Splits input into two parts and applies learned transformations to adjust scale and translation, allowing for more flexible distributions.
- **Permutation Layers**: Helps to better capture dependencies between features by reordering them during the flow.
- **Simulation-to-Data Mapping**: Transforms MC-generated data to match real-world experimental data.

### Data Location in Rivanna
- `/ptgroup/Forhad/CNF_DATA`

### Python Libraries and Versions

The following Python libraries are used in this project:

- **numpy**: 2.0.2
- **torch**: 2.6.0
- **matplotlib**: 3.9.4
- **tqdm**: 4.67.1
- **scikit-learn**: 1.6.1
