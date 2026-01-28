# ARCR - Action-Relevant Causal Representation for EV Energy Recovery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ARCR** (Action-Relevant Causal Representation) is a novel framework for industrial-grade energy recovery in electric vehicles, implementing causal representation learning with intervention consistency and structured latent dynamics.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

This repository contains the complete implementation of the **Action-Relevant Causal Representation (ARCR)** framework for electric vehicle energy recovery systems. The framework combines:

1. **Intervention-Consistent Causal World Models** that distinguish observational and interventional distributions
2. **Action-Relevant Causal Representation Learning** that identifies action-relevant state dimensions
3. **Structured Latent Dynamics** with causal Markov properties
4. **Causally-Constrained Maximum Entropy Actor-Critic** for policy optimization

## Key Features

- **Causal Representation Learning**: Learns compact representations that isolate action-relevant causal factors
- **Intervention Consistency**: Maintains consistency under counterfactual interventions
- **High-Fidelity Simulation**: Physics-based EV simulator with validated dynamics
- **Real-time Inference**: < 2ms inference time on GPU
- **Robustness Testing**: Comprehensive distribution shift experiments
- **Causal Structure Analysis**: Interpretable causal masks and transition matrices

## Architecture

The ARCR framework consists of four core components:
