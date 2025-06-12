#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import math
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import newton
import warnings
warnings.simplefilter("ignore")

# Define a hardcoded password (for example)
PASSWORD = "5555"

# Use session state to keep track of authentication status
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Ask user for password
    password_input = st.text_input("Enter password:", type="password")
    if password_input == PASSWORD:
        st.session_state["authenticated"] = True
        st.success("Password accepted! Welcome to the app.")
    else:
        st.error("Please enter the correct password.")
        st.stop()

# Proceed with the main app content after login
st.write("Welcome to the rest of your app!")

df1=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx", sheet_name='Sheet3')
df2=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx", sheet_name='New')
df3=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx", sheet_name='Cooling water')
df4=pd.read_excel(r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx", sheet_name='Trial and Error method')

df1['Date'] = df1['Date'].astype(str)

e = st.sidebar.slider("Experiment number : ", max_value = 100, min_value = 0, value = 53, step = 1) 
T_gin = st.sidebar.slider('Humid air inlet temperature', value=145, min_value=0, max_value=250, step=1)
T_cout = st.sidebar.slider('Cooling water outlet temperature : ', value=50, min_value=0, max_value=100, step=1)
steam_flowrate = st.sidebar.slider('Vapour flow rate : ', value=29, min_value=0, max_value=200, step=1)
Air_flowrate =  st.sidebar.slider('Air flow rate : ', value=106, min_value=0, max_value=500, step=1)
a = st.sidebar.slider('Wall temperature coefficient : ', value=0.62, min_value=0.0, max_value=2.02, step=0.01)
CW_flowrate =  st.sidebar.slider('Coling water flow rate : ', value=125, min_value=0, max_value=2000, step=1)
n =  st.sidebar.slider('Number of segments of the experiments : ', value=8, min_value=8, max_value=100, step=1)
st.title("Condensing heat exchanger (Experiment VS Calculation)")
st.write(f"""This program applies forward differencing to a counter-flow serpentine condensing heat exchanger.
The input conditions are:
- The inlet temperature of the humid air
- The inlet temperature of the cooling water
- The mass flow rates of both fluids
- The mass fraction of vapor in the humid air

In the first iteration, experimental data is used for the average values of both the hot and cold fluids. Subsequently, the average values are the mean values calculated over all iterations.
Additionally, the wall temperature in the first iteration is assumed using a multiplier, which can be adjusted via a slider. In subsequent iterations, the wall temperature is calculated.""")


# In[2]:


img = Image.open('Picture1.png')
st.image(img, caption = 'Illustration of the finite difference analysis', width = 800, channels = 'RGB')


# In[3]:

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8']]

Wall_temp = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']]

dew_point = df1.loc[e,['Dew point_1','Dew point_2','Dew point_3','Dew point_4','Dew point_5','Dew point_6',
'Dew point_7','Dew point_8']]

Cooling_water = df1.loc[e ,['Temperature increase of the cooling water_1',
 'Temperature increase of the cooling water_2',
 'Temperature increase of the cooling water_3',
 'Temperature increase of the cooling water_4',
 'Temperature increase of the cooling water_5',
 'Temperature increase of the cooling water_6',
 'Temperature increase of the cooling water_7',
 'Temperature increase of the cooling water_8',
'Temperature increase of the cooling water_9']].values

Flue_gas = df1.loc[e ,['Temperature decrease of the mixture_1',
 'Temperature decrease of the mixture_2',
 'Temperature decrease of the mixture_3',
 'Temperature decrease of the mixture_4',
 'Temperature decrease of the mixture_5',
 'Temperature decrease of the mixture_6',
 'Temperature decrease of the mixture_7',
 'Temperature decrease of the mixture_8',
'Temperature decrease of the mixture_9']].values

Wall_temperature1 = df1.loc[e ,['Tube_coil_surface_temp1',
 'Tube_coil_surface_temp2',
 'Tube_coil_surface_temp3',
 'Tube_coil_surface_temp4',
 'Tube_coil_surface_temp5',
 'Tube_coil_surface_temp6',
 'Tube_coil_surface_temp7',
 'Tube_coil_surface_temp8']].values

# Constants that don't change during iterations
M_h2o = 18.015
M_g = 28.96
D_i = 0.014  # m
D_o = 0.018
p = 101325  # Pa
R_air = 8.314
R_water = 8.314
a_antoine = 16.262
b_antoine = 3799.89
c_antoine = 226.35
p_tot = p / 1000


# Predefined interpolation data (could be moved to config files)
# Water density data
density_temp = [0.1, 1, 4, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                85, 90, 95, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                320, 340, 360, 373.946]
density_values = [999.85, 999.9, 999.97, 999.7, 999.1, 998.21, 997.05, 995.65, 994.03, 992.22,
                  990.21, 988.04, 985.69, 983.2, 980.55, 977.76, 974.84, 971.79, 968.61, 965.31,
                  961.89, 958.35, 950.95, 943.11, 926.13, 907.45, 887, 864.66, 840.22, 813.37, 783.63,
                  750.28, 712.14, 667.09, 610.67, 527.59, 322]

# Saturation pressure data
pressure = [0.15, 0.2, 0.25, 0.5]
temperature = [53.983, 60.073, 64.980, 81.339]

# Thermal conductivity data for water
k_water_temp = [0.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.6]
k_water_values = [555.75, 578.64, 598.03, 614.5, 628.56, 640.6, 650.91, 659.69, 667.02, 672.88, 677.03]

# Specific heat data for water
cp_water_temp = [0.01, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
cp_water_values = [4.2199, 4.1955, 4.1844, 4.1816, 4.1801, 4.1796, 4.1815, 4.1851, 4.1902, 4.1969, 4.2053, 4.2157]

# Viscocity data for water
u_water_temp = [17.51, 24.1, 28.98, 32.9, 36.18, 39.02, 41.53, 43.79, 45.83, 60.09, 69.13, 75.89, 81.35, 85.95, 89.96,
                           93.51, 96.71, 99.63, 102.32, 104.81, 107.13, 109.32, 111.37, 111.37, 113.32, 115.17, 116.93,
                           118.62, 120.23, 123.27, 126.09, 128.73, 131.2, 133.54, 138.87, 143.63, 147.92, 151.85,
                           155.47, 158.84, 161.99, 164.96, 167.76, 170.42, 172.94, 175.36, 177.67, 179.88, 184.06,
                           187.96, 191.6, 195.04, 198.28, 201.37, 204.3, 207.11, 209.79, 212.37, 214.85, 217.24,
                           219.55, 221.78, 223.94, 226.03, 228.06, 230.04, 231.96, 233.84]
u_water_values = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000011, 0.000011,
                           0.000011, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000013,
                           0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013,
                           0.000013, 0.000013, 0.000013, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014,
                           0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015,
                           0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016,
                           0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017]

# Air properties data
cp_air_temp = [0, 6.9, 15.6, 26.9, 46.9, 66.9, 86.9, 107, 127, 227, 327, 427, 527, 627]
cp_air_values = [1.006, 1.006, 1.006, 1.006, 1.007, 1.009, 1.01, 1.012, 1.014, 1.03, 1.051, 1.075, 1.099, 1.121]

# Viscosity data
visc_air_temp = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 225, 300]
visc_air_values = [0.00001715, 0.0000174, 0.00001764, 0.00001789, 0.00001813, 0.00001837, 
                   0.0000186, 0.00001907, 0.00001953, 0.00001999, 0.00002088, 0.00002174, 
                   0.00002279, 0.0000238, 0.00002478, 0.00002573, 0.00002666, 0.00002928]

# Thermal conductivity data for air
k_air_temp = [-190, -150, -100, -75, -50, -25, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150,
              175, 200, 225, 300, 412, 500, 600, 700, 800, 900, 1000, 1100]
k_air_values = [7.82, 11.69, 16.2, 18.34, 20.41, 22.41, 23.2, 23.59, 23.97, 24.36, 24.74, 25.12, 25.5, 25.87, 26.24,
                26.62, 27.35, 28.08, 28.8, 30.23, 31.62, 33.33, 35, 36.64, 38.25, 39.83, 44.41, 50.92, 55.79, 61.14,
                66.32, 71.35, 76.26, 81.08, 85.83]

# Thermal conductivity of vapor
k_g_vapour_temp = [0.01, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                      110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360]
k_g_vapour_values = [0.0171, 0.0173, 0.0176, 0.0179, 0.0182, 0.0186, 0.0189, 0.0192, 0.0196, 0.02, 0.0204, 
                      0.0208, 0.0212, 0.0216, 0.0221, 0.0225, 0.023, 0.0235, 0.024, 0.0246, 0.0251, 0.0262, 
                      0.0275, 0.0288, 0.0301, 0.0316, 0.0331, 0.0347, 0.0364, 0.0382, 0.0401, 0.0442, 0.0487, 
                      0.054, 0.0605, 0.0695, 0.0836, 0.11, 0.178]

# Specific heat of vapor
cp_vapour_temp = [-23, 2, 27, 52, 77, 102, 127, 177, 227, 277, 327, 377, 427, 477, 527, 577, 627, 677, 727] 
cp_vapour_values = [1.855, 1.859, 1.864, 1.871, 1.88, 1.89, 1.901, 1.926, 1.954, 1.984, 2.015, 2.047, 2.08, 2.113, 2.147, 2.182, 2.217, 2.252, 2.288] 


# Nusselt number data for laminar flow
x_values = [0.001, 0.004, 0.01, 0.04, 0.08, 0.1, 0.2]
M_Nu = [19.29, 12.09, 8.92, 5.81, 4.86, 4.64, 4.15]
L_Nu = [12.8, 8.03, 6, 4.17, 3.77, 3.71, 3.66]

def calculate_water_properties(T_c):
    """Calculate all water properties at once for a given temperature."""
    rho_c = np.interp(T_c, density_temp, density_values)
    k_c = np.interp(T_c, k_water_temp, k_water_values) / 1000
    c_pc = np.interp(T_c, cp_water_temp, cp_water_values) * 1000
    return rho_c, k_c, c_pc

def calculate_air_properties(T_g, y_h2o):
    """Calculate all air properties at once for a given temperature and water mole fraction."""
    cp_Air = np.interp(T_g, cp_air_temp, cp_air_values)
    u_air = np.interp(T_g, visc_air_temp, visc_air_values)
    k_g_air = np.interp(T_g, k_air_temp, k_air_values) / 1000
    cp_vapour = np.interp(T_g,cp_vapour_temp,cp_vapour_values)
    
    y_air = 1 - y_h2o
    M_m = y_air * M_g + y_h2o * M_h2o
    c_pg = ((M_g/M_m) * y_air * cp_Air + (M_h2o/M_m) * y_h2o * cp_vapour)
    
    return cp_Air, u_air, k_g_air, c_pg

def calculate_interface_equation(T_i, T_g, h_g, h_fg, y_h2o, h_c, T_c, alpha_g, D_h2oair, c_pg):
    """Calculate the interface equation for Newton's method."""
    y_i = np.exp(a_antoine - (b_antoine / (T_i + c_antoine))) / p_tot
    y_ni = 1 - y_i
    y_nb = 1 - y_h2o
    y_lm = (y_ni - y_nb) / math.log(y_ni / y_nb)
    Le_h20air = alpha_g / D_h2oair
    k_m = (h_g * M_h2o) / (c_pg * 1000 * M_g * y_lm * Le_h20air ** (2/3))
    return ((h_g * T_g + h_fg * 1000 * k_m * (y_h2o - y_i) + h_c * T_c) / (h_g + h_c)) - T_i

def main_loop(n, m_frac, T_cout, T_gin, CW_flowrate, steam_flowrate, m_g, a): 
    # Initialize lists to store results
    y_H2o = []
    Sat_temp = []
    Water_density = []
    Mass_flowrate = []
    Water_velocity = []
    Water_Dynamic_viscosity = []
    Water_Reynolds = []
    Water_thermal_conductivity = []
    Water_specific_heat = []
    Water_Nusselt_number = []
    Water_heat_Transfer_coefficient = []
    Wall_temperature2 = []
    Op_temp_air = []
    Inlet_temp_air = []
    Density_air = []
    FlowRate_air = []
    Velocity_air = []
    Specific_heat_air = []
    Reynolds_air = []
    Thermal_conductivity_air = []
    Thermal_diffusivity_air = []
    Prandtl = []
    Nusselt_air = []
    Heat_transfer_air = []
    Latent_heat_air = []
    Mass_of_diffusivity = []
    Lewis_air = []
    Temperature_interface = []
    Vapour_mole_interface = []
    Mass_transfer_coefficient_air = []
    numbering = []
    Logarithmic_mole_average = []
    Outlet_temp_air = []
    Inlet_temp_water = []
    Condensation_rate = []    
    M_frac = steam_flowrate/(steam_flowrate+Air_flowrate)
    m_g = steam_flowrate+Air_flowrate
    
    for i in range(n):
        # 1. Calculate water mole fraction
        y_h2o = (float(M_frac) / M_h2o) / ((float(M_frac) / M_h2o) + ((1 - float(M_frac)) / M_g))
        y_H2o.append(y_h2o)
        
        # 2. Calculate saturation temperature
        P_w = y_h2o * 1  # Atm
        T_sat = np.interp(P_w, pressure, temperature)
        Sat_temp.append(T_sat)
        
        # 3. Handle temperatures
        if i == 0:
            T_c = T_cout
            T_g = T_gin
            Inlet_temp_water.append(T_cout)
            Outlet_temp_air.append(T_gin)
        else:
            T_c = Inlet_temp_water[-1]
            T_g = Outlet_temp_air[-1]

        # 4. Calculate water properties
        rho_c, k_c, c_pc = calculate_water_properties(T_c)
        Water_density.append(rho_c)
        
        # Calculate water flow properties
        m_c = (CW_flowrate * rho_c) / (3600 * 1000 * 3)  # Kg/s
        Mass_flowrate.append(m_c)
        
        A_c = (((D_i**2 * math.pi) / 4) * 8) / n
        v_c = m_c / (rho_c * A_c)
        Water_velocity.append(v_c)
        
        # Calculate dynamic viscosity
        T = T_c + 273
        u_c = (0.0000000000330128 * T**4 - 0.0000000448315 * T**3 + 
               0.0000228887 * T**2 - 0.00521331 * T + 0.447905)
        Water_Dynamic_viscosity.append(u_c)
        
        # Calculate Reynolds number
        Re_c = (rho_c * v_c * D_i) / u_c
        Water_Reynolds.append(Re_c)
        
        Water_thermal_conductivity.append(k_c)
        Water_specific_heat.append(c_pc)
        
        alpha_c = k_c / (rho_c * c_pc)
        pr_c = (u_c / rho_c) / alpha_c
        
        # Calculate Nusselt number and heat transfer coefficient
        if Re_c < 3000:
            x = 0.30
            x_ = ((2 * x) / D_i) / (Re_c * pr_c)
            Mean_Nu = np.interp(x_, x_values, M_Nu)
            h_c = (Mean_Nu * k_c) / D_i
        else:
            if Re_c <= 2e4:
                f = 0.316 * Re_c ** -0.25
            else:
                f = (0.790 * math.log(Re_c) - 1.64) ** -2
            Nu_c = (f/8) * (Re_c - 1000) * pr_c / (1 + 12.7 * (f/8)**0.5 * (pr_c**(2/3) - 1))
            h_c = (Nu_c * k_c) / D_i
        
        Water_Nusselt_number.append(Nu_c if Re_c >= 3000 else Mean_Nu)
        Water_heat_Transfer_coefficient.append(h_c)
        
        # Wall temperature
        if i == 0:
            T_w = a * (T_gin - T_cout)
            # T_w = Wall_temperature1[i]
            Wall_temperature2.append(T_w)
        
        # 5. Calculate air properties
        T_g_float = float(T_g)
        cp_Air, u_air, k_g_air, c_pg = calculate_air_properties(T_g_float, y_h2o)
        
        # Calculate mixture properties
        y_air = 1 - y_h2o
        T = T_g_float + 273
        rho_air = ((p * M_g / 1000) / (R_air * T)) * y_air
        rho_water = ((p * M_h2o / 1000) / (R_water * T)) * y_h2o
        rho_g = rho_water + rho_air
        Density_air.append(rho_g)
        st.write(f"Temperature at which density calculated {Outlet_temp_air[i]}")
        st.write(f"Temperature  {T}")
        st.write(f"Air density : {rho_g}")
        
        m_g = (steam_flowrate + Air_flowrate)/60**2- np.sum(Condensation_rate)
        FlowRate_air.append(m_g)
        st.write(f"Mass flow rate: {m_g}")
        A_gap = (0.011 * 8) / n
        v_g = m_g / (rho_g * A_gap)
        Velocity_air.append(v_g)
        st.write(f"Air velocity: {v_g}")
        Specific_heat_air.append(c_pg)
        
        # Calculate viscosity using Wilke's method
        u_water = np.interp(T_g_float, u_water_temp, u_water_values)
        
        Q_av = (math.sqrt(2) / 4) * (1 + (M_g / M_h2o)) ** -0.5 * ((1 + math.sqrt(u_air / u_water)) * (M_h2o / M_g) ** 0.25) ** 2
        Q_va = (math.sqrt(2) / 4) * (1 + (M_h2o / M_g)) ** -0.5 * ((1 + math.sqrt(u_water / u_air)) * (M_g / M_h2o) ** 0.25) ** 2
        u_g = ((y_air * u_air) / (y_air + y_h2o * Q_av)) + ((y_h2o * u_water) / (y_h2o + y_air * Q_va))
        
        Re_g = (rho_g * v_g * D_o) / u_g
        Reynolds_air.append(Re_g)
        st.write(f"Reynolds number : {Re_g}")
        
        # Thermal conductivity of vapor
        k_g_vapour = np.interp(T_g_float, k_g_vapour_temp, k_g_vapour_values)
        
        k_g = ((y_air * k_g_air) / (y_air + Q_av * y_h2o)) + ((y_h2o * k_g_vapour) / (y_h2o + y_air * Q_va))
        Thermal_conductivity_air.append(k_g)
        
        alpha_g = k_g / (rho_g * c_pg * 1000)
        Thermal_diffusivity_air.append(alpha_g)
        
        pr_g = (u_g / rho_g) / alpha_g
        Prandtl.append(pr_g)

        # Calculate Nusselt number for air
        if (Re_g <= 2*10**6) and (Re_g >= 1000) and (0.7 <= np.round(pr_g,1)) and (np.round(pr_g,1) <= 500):
            c = 0.27 # 
            m = 0.63 # 
            Nu_g = c*(Re_g**m)*(pr_g**0.36)
            Nusselt_air.append(Nu_g)
            print("Nusselt number for the air side:", np.round(Nu_g,4))
    
            h_g = (Nu_g * k_g)/D_o
            Heat_transfer_air.append(h_g)
            
            # Latent heat
            h_fg = -0.0021 * T_g_float**2 - 2.2115 * T_g_float + 2499
            Latent_heat_air.append(h_fg)
            
            # Mass diffusivity
            D_h2oair = (6.057e-6 + 4.055e-8 * T + 1.25e-10 * T**2 - 3.367e-14 * T**3)
            Mass_of_diffusivity.append(D_h2oair)
            
            D_h2og = D_h2oair * (alpha_g / (k_g_air / (rho_air * cp_Air * 1000)))
            
            # Lewis number
            Le_h20air = alpha_g / D_h2oair
            Lewis_air.append(Le_h20air)
            
        # Interface temperature calculation
        if T_w < T_sat:
            try:
                T_i_solution = newton(
                    calculate_interface_equation, 
                    60,  # Initial guess
                    args=(T_g, h_g, h_fg, y_h2o, h_c, T_c, alpha_g, D_h2oair, c_pg)
                )
                Temperature_interface.append(T_i_solution)
                
                y_i = np.exp(a_antoine - (b_antoine / (T_i_solution + c_antoine))) / p_tot
                Vapour_mole_interface.append(y_i)
                
                y_ni = 1 - y_i
                y_nb = 1 - y_h2o
                y_lm = (y_ni - y_nb) / math.log(y_ni / y_nb)
                Logarithmic_mole_average.append(y_lm)
                
                k_m = (h_g * M_h2o) / (c_pg * 1000 * M_g * y_lm * Le_h20air ** (2/3))
                Mass_transfer_coefficient_air.append(k_m)
                numbering.append(i + 1)
            except:
                # Handle convergence error
                Temperature_interface.append(np.nan)
                Vapour_mole_interface.append(np.nan)
                Logarithmic_mole_average.append(np.nan)
                Mass_transfer_coefficient_air.append(np.nan)
                numbering.append(i + 1)
        
        # Outlet temperature calculations
        delta_Ai = (0.0206 * 8) / n
    
        if T_w < T_sat :
            T_gout = ((m_g * c_pg * 1000 - (h_g/2) * delta_Ai) * T_gin + h_g * delta_Ai * T_i_solution) / \
                    (m_g * c_pg * 1000 + (h_g/2) * delta_Ai)
        else:
            T_gout = ((m_g * c_pg * 1000 - (h_g/2) * delta_Ai) * T_gin + h_g * delta_Ai * T_w) / \
                    (m_g * c_pg * 1000 + (h_g/2) * delta_Ai)
    
        Outlet_temp_air.append(T_gout)
        st.write(f"Outlet_temp_air {Outlet_temp_air}")
        # Inlet temperature calculations
        if T_w < T_sat:
            T_cin = T_cout - ((h_g * (T_gin - T_i_solution) * delta_Ai + 
                              h_fg * k_m * (y_h2o - y_i) * delta_Ai) / 
                             (m_c * c_pc))
        else:
            T_cin = T_cout - ((h_g * (T_gin - T_w) * delta_Ai) / (m_c * c_pc))
    
        Inlet_temp_water.append(T_cin)
    
        # Condensation rate
        if T_w < T_sat:
            m_cd = k_m * (y_h2o - y_i) * delta_Ai
            Condensation_rate.append(m_cd)
            M_frac = (steam_flowrate - np.sum(Condensation_rate)) / (m_g - np.sum(Condensation_rate))
        else:
            Condensation_rate.append(0)
        
        # Wall temperature for subsequent iterations
        if i != 0:
            numerator = m_c * c_pc * (Inlet_temp_water[i-1] - Inlet_temp_water[i]) * 3
            denominator = h_c * delta_Ai * 3
            T_w = T_c + (numerator / denominator)
            # T_w = Wall_temperature1[i]
            Wall_temperature2.append(T_w)
    
    # Debug print
    # st.write(f"Segment {i+1}: Gas {T_gin:.1f}→{T_gout:.1f}°C, Water {T_cout:.1f}→{T_c_in:.1f}°C, Wall {T_w:.1f}°C")
    # st.write(f"len Outlet_temp_air: {len(Outlet_temp_air)}, len Inlet_temp_water: {len(Inlet_temp_water)}")

    
    # Return all the calculated lists
    results =  {
        'y_H2o': y_H2o,
        'Sat_temp': Sat_temp,
        'Water_density': Water_density,
        'Mass_flowrate': Mass_flowrate,
        'Water_velocity': Water_velocity,
        'Water_Dynamic_viscosity': Water_Dynamic_viscosity,
        'Water_Reynolds': Water_Reynolds,
        'Water_thermal_conductivity': Water_thermal_conductivity,
        'Water_specific_heat': Water_specific_heat,
        'Water_Nusselt_number': Water_Nusselt_number,
        'Water_heat_Transfer_coefficient': Water_heat_Transfer_coefficient,
        'Wall_temperature2': Wall_temperature2,
        'Op_temp_air': Op_temp_air,
        'Inlet_temp_air': Inlet_temp_air,
        'Density_air': Density_air,
        'FlowRate_air': FlowRate_air,
        'Velocity_air': Velocity_air,
        'Specific_heat_air': Specific_heat_air,
        'Reynolds_air': Reynolds_air,
        'Thermal_conductivity_air': Thermal_conductivity_air,
        'Thermal_diffusivity_air': Thermal_diffusivity_air,
        'Prandtl': Prandtl,
        'Nusselt_air': Nusselt_air,
        'Heat_transfer_air': Heat_transfer_air,
        'Latent_heat_air': Latent_heat_air,
        'Mass_of_diffusivity': Mass_of_diffusivity,
        'Lewis_air': Lewis_air,
        'Temperature_interface': Temperature_interface,
        'Vapour_mole_interface': Vapour_mole_interface,
        'Mass_transfer_coefficient_air': Mass_transfer_coefficient_air,
        'numbering': numbering,
        'Logarithmic_mole_average': Logarithmic_mole_average,
        'Outlet_temp_air': Outlet_temp_air,
        'Inlet_temp_water': Inlet_temp_water,
        'Condensation_rate': Condensation_rate
    }
    return results        
def average_list(values, num_averages):
# Determine the number of values per average
    values_per_avg = len(values) // num_averages
    
     #Calculate the averages
    averaged_values = [
        sum(values[i * values_per_avg: (i + 1) * values_per_avg]) / values_per_avg
        for i in range(num_averages)
    ]
    
    # Handle the remainder if the number of values is not perfectly divisible
    remainder = len(values) % num_averages
    if remainder:
        averaged_values[-1] = (
            sum(values[-remainder:]) / remainder
        )
    
    return averaged_values

results = main_loop(n, steam_flowrate/(steam_flowrate + Air_flowrate), T_cout, T_gin, CW_flowrate,steam_flowrate, (steam_flowrate + Air_flowrate),a)

# # Example lists to process
#Wall_temperature2 = average_list(Wall_temperature2, 8)
#Outlet_temp_air = average_list(Outlet_temp_air, 8)
#Inlet_temp_water = average_list(Inlet_temp_water, 8)
#Temperature_interface = average_list(Temperature_interface, 8)
#Sat_temp = average_list(Sat_temp, 8)
########################T_cout - ((h_g*(T_g-T_i_solution)*delta_Ai + h_fg*k_m*(y_h2o - y_i)*delta_Ai) / (m_c*c_pc))
#print("Temperature_interface",Temperature_interface)
# print("Inlet_temp_water",Inlet_temp_water)

cc = np.sum(results['Condensation_rate'])*1000
condd = df1.loc[e ,['First_Cond','Second_Cond','Third_Cond','Fourth_Cond','Fifth_Cond','Sixth_Cond','Seventh_Cond','Eighth_Cond']]/(df1.loc[e,'Time']*1000)
# condd.sum()*1000
condensation = pd.DataFrame({"Type":["Calculated",'Experimental'],
                         "Values":[cc,condd.sum()*1000]})
# Parameters for the experiment
experiment_parameters = f"""
Experiment's parameters are :

T_gin = {df1.loc[e,'Mixture tin, oC']}°C,

Vapour flow rate = {df1.loc[e,'Vapour flow rate, kg/h']} kg/h,

Inlet temperature of humid air = {df1.loc[e,'Mixture tin, oC']}°C,

Mixture flow rate = {df1.loc[e,'Mixture  (air+vapour) flow rate, kg/h']} kg/h,

Air flow rate = {df1.loc[e,'Mixture  (air+vapour) flow rate, kg/h'] - df1.loc[e,'Vapour flow rate, kg/h']} kg/h

Cooling water flow rate = {df1.loc[e,'Cooling water flow rate, l/h']} l/h,

Reynolds number = {df1.loc[e,'Re']} and

Mass fraction of water vapour = {df1.loc[e,'Mass Fraction']} %
"""

st.write(f"results['Outlet_temp_air'] : {results['Outlet_temp_air']}\n")
st.write(f"results['Inlet_temp_water'] : {results['Inlet_temp_water']}")

# Create a bar plot for condensation data

def plot_results(results):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), constrained_layout=True)

    # 1. Temperatures
    for key, label in [
        ('Wall_temperature2', 'Wall Temperature'),
        ('Op_temp_air', 'Operational Air Temp'),
        ('Inlet_temp_air', 'Inlet Air Temp'),
        ('Outlet_temp_air', 'Outlet Air Temp'),
        ('Inlet_temp_water', 'Inlet Water Temp'),
        ('Temperature_interface', 'Interface Temperature')
    ]:
        y = results[key]
        x = np.arange(1, len(y)+1)
        axs[0].plot(x, y, label=label)
    axs[0].set_title('Temperature Profiles (°C)')
    axs[0].set_xlabel('Measurement Point')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Flow & velocity
    for key, label in [
        ('Mass_flowrate', 'Mass Flowrate'),
        ('Water_velocity', 'Water Velocity'),
        ('Velocity_air', 'Air Velocity'),
        ('FlowRate_air', 'Air Flow Rate')
    ]:
        y = results[key]
        x = np.arange(1, len(y)+1)
        axs[1].plot(x, y, label=label)
    axs[1].set_title('Flow and Velocity')
    axs[1].set_xlabel('Measurement Point')
    axs[1].set_ylabel('Flow/Velocity')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Heat transfer and physical properties
    for key, label in [
        ('Water_heat_Transfer_coefficient', 'Water Heat Transfer Coeff.'),
        ('Heat_transfer_air', 'Air Heat Transfer'),
        ('Latent_heat_air', 'Latent Heat Air'),
        ('Water_Nusselt_number', 'Water Nusselt Number'),
        ('Nusselt_air', 'Air Nusselt Number')
    ]:
        y = results[key]
        x = np.arange(1, len(y)+1)
        axs[2].plot(x, y, label=label)
    axs[2].set_title('Heat Transfer Properties')
    axs[2].set_xlabel('Measurement Point')
    axs[2].set_ylabel('Heat Transfer / Coefficients')
    axs[2].legend()
    axs[2].grid(True)

    # 4. Miscellaneous
    for key, label in [
        ('Water_Reynolds', 'Water Reynolds'),
        ('Reynolds_air', 'Air Reynolds'),
        ('Mass_of_diffusivity', 'Mass of Diffusivity'),
        ('Condensation_rate', 'Condensation Rate')
    ]:
        y = results[key]
        x = np.arange(1, len(y)+1)
        axs[3].plot(x, y, label=label)
    axs[3].set_title('Miscellaneous Parameters')
    axs[3].set_xlabel('Measurement Point')
    axs[3].set_ylabel('Values')
    axs[3].legend()
    axs[3].grid(True)

    st.pyplot(fig)
plot_results(results)

# Display experiment parameters below the plots
st.text(experiment_parameters)
# st.text(Wall_temperature2)




# In[ ]:





# In[ ]:




