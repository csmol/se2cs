# TimeLIME: Transparent Interpretation of Model Explanations

## Introduction

TimeLIME is a tool designed to provide transparent interpretation of model predictions by generating explanations for time-series data. It helps users understand the reasoning behind a model's predictions, particularly in the context of time-varying data. Here is the link to the publication of the applicaition of [TimeLIME](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9371412)

## Getting Started

To run TimeLIME on your own data, follow these simple steps:

### Step 1: Prepare Your Data

1. Place your time-series data files in the `data` folder.
2. Ensure that your data files are in the CSV format and contain the necessary features for analysis.

### Step 2: Set Target Goal

1. Update the `target/target.csv` file with the target goal you want to analyze.

### Step 3: Run the Experiment

1. Execute the `runexp.py` script using the following command:
   ```bash
   python runexp.py
