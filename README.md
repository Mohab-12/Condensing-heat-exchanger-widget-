1. Overview

This application is an interactive numerical model of a counter-flow condensing heat exchanger, implemented in Python and deployed using Streamlit.

The model simulates heat and mass transfer with condensation along the heat exchanger using a segment-by-segment approach, allowing users to:

Predict outlet temperatures
Estimate condensation rates
Analyse local heat transfer behaviour
Compare model results with experimental data
2. Key Features
Segment-wise simulation (default: 40 segments)
Built-in thermophysical properties via CoolProp
Automatic convergence on cooling water inlet temperature
Direct comparison with experimental data
Interactive visualization dashboard
Exportable results (CSV)
3. Input Data Requirements

The model requires an Excel file (.xlsx) with a sheet named:

Sheet1
Required Columns

Your dataset must include the following columns:

Temperature Profiles
Temperature decrease of the mixture_1 → _9
Temperature increase of the cooling water_1 → _9
Tube_coil_surface_temp1 → Tube_coil_surface_temp8
Flow Rates
Vapour flow rate, kg/h
Cooling water flow rate, l/h
Mixture (air+vapour) flow rate, kg/h
Condensation flow rate(Kg/min)

Column names must match exactly, otherwise the model will fail.

4. How to Run the App
Step 1: Launch the Application

Run the script:

streamlit run Streamlit_widget.py
Step 2: Upload Data (Optional)
Use the sidebar to upload your Excel file
If no file is uploaded, the app uses the default dataset (Data.xlsx)
Step 3: Select Experiment
Choose the Experiment ID (row index in Excel)
Each row represents one experimental case
Step 4: Run Simulation

Click: Run Model

The model will:

Load the selected experiment
Perform iterative convergence
Solve segment-wise heat and mass transfer
Display results
5. Output Dashboard
5.1 Main Summary
Outlet air temperature
Cooling water inlet temperature
Total condensed mass
5.2 Temperature Profiles
Model vs experimental comparison
Air and cooling water temperatures along segments
5.3 Heat Transfer Coefficients
Air-side heat transfer coefficient
Water-side heat transfer coefficient
5.4 Condensation Analysis
Total condensation (model vs experiment)
Cumulative condensation along the exchanger
5.5 Flow Parameters
Reynolds numbers (air & water)
Flow regime insights
5.6 Energy Balance
Sensible heat
Latent heat (condensation)
Water-side heat transfer
Energy imbalance (model accuracy indicator)
5.7 Raw Data
Full segment-wise results table
Downloadable as CSV

6. Model Description (Simplified)

The heat exchanger is divided into N segments.

For each segment:

Thermophysical properties are calculated
Heat transfer coefficients are evaluated
Condensation is checked via dew point
Energy and mass balances are solved
Temperatures and condensation rates are updated

The model iterates until:

Cooling water outlet ≈ Experimental value

7. Customization

The model is fully customizable:

You can modify:
Heat exchanger geometry
Number of segments
Fluid properties correlations
Input dataset structure
Important note:

If you use a new dataset:

You must adapt column names in get_experiment_data()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19677977.svg)](https://doi.org/10.5281/zenodo.19677977)
