import numpy as np
import pandas as pd
import math
from scipy.optimize import newton
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
import warnings
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Configure settings
warnings.simplefilter("ignore")

# Constants
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
pressure = [0.15, 0.2, 0.25, 0.5]
temperature = [53.983, 60.073, 64.980, 81.339]

# Nusselt number data for laminar flow
x_values = [0.001, 0.004, 0.01, 0.04, 0.08, 0.1, 0.2]
M_Nu = [19.29, 12.09, 8.92, 5.81, 4.86, 4.64, 4.15]
L_Nu = [12.8, 8.03, 6, 4.17, 3.77, 3.71, 3.66]

def load_data():
    file_path = r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx"
    df1 = pd.read_excel(file_path, sheet_name='Sheet3')
    df1['Date'] = df1['Date'].astype(str)
    return df1

def get_experiment_data(df, e):
    data = {
        'Humid_air': df.loc[e, [f'Temperature decrease of the mixture_{i}' for i in range(1,10)]].values,
        'Cooling_water': df.loc[e, [f'Temperature increase of the cooling water_{i}' for i in range(1,10)]].values,
        'Wall_temp': df.loc[e, [f'Tube_coil_surface_temp{i}' for i in range(1,9)]].values,
        'dew_point': df.loc[e, [f'Dew point_{i}' for i in range(1,9)]].values,
        'steam_flowrate': df.loc[e, 'Vapour flow rate, kg/h']/3600,
        'CW_flowrate': df.loc[e, 'Cooling water flow rate, l/h'],
        'Mixture_flowrate': df.loc[e, 'Mixture  (air+vapour) flow rate, kg/h']/3600,
        'exp_cond_rate': df.loc[e, 'Condensation flow rate(Kg/min)']/60 }
    return data

def calculate_water_properties(T_c):
    """Vectorized water properties calculation"""
    T_k = T_c + 273.1
    return (
        PropsSI("D", "T", T_k, "P", 101325, "water"),
        PropsSI("CONDUCTIVITY", "T", T_k, "P", 101325, "Water"),
        PropsSI("C", "T", T_k, "P", 101325, "Water"),
        PropsSI("V", "T", T_k, "P", 101325, "water")
    )

def calculate_air_properties(T_g, m_frac):
    """Vectorized air properties calculation"""
    T_k = T_g + 273.1
    return (
        HAPropsSI("mu", "T", T_k, "P", 101325, "W", m_frac),
        HAPropsSI("k", "T", T_k, "P", 101325, "W", m_frac),
        HAPropsSI("Cha", "T", T_k, "P", 101325, "W", m_frac)
    )

def calculate_nusselt_water(Re_c, pr_c, k_c, D_i, x_values=x_values, M_Nu=M_Nu):
    """Calculate Nusselt number and heat transfer coefficient for water"""
    if Re_c < 3000:  # Laminar flow
        x = 0.30  # Length in meters
        x_ = ((2 * x) / D_i) / (Re_c * pr_c)
        x_ = np.clip(x_, min(x_values), max(x_values))
        Mean_Nu = np.interp(x_, x_values, M_Nu)
        h = (Mean_Nu * k_c) / D_i
        return Mean_Nu, h
    else:  # Turbulent flow
        if Re_c <= 2e4:
            f = 0.316 * Re_c ** -0.25
        else:
            f = (0.790 * math.log(Re_c) - 1.64) ** -2
        Nu_c = (f / 8) * (Re_c - 1000) * pr_c / (1 + 12.7 * (f / 8) ** 0.5 * (pr_c ** (2 / 3) - 1))
        h = (Nu_c * k_c) / D_i
        return Nu_c, h

def calculate_nusselt_air(Re_g, pr_g, k_g):
    """Calculate Nusselt number for air"""
    if 1000 <= Re_g <= 2e6 and 0.7 <= round(pr_g, 1) <= 500:
        return 0.27 * (Re_g**0.63) * (pr_g**0.36)
    return 10  # Fallback value

def run_simulation(e, T_gin, T_cout, steam_flowrate, Air_flowrate, a, n):
    df = load_data()
    data = get_experiment_data(df, e)
    
    # Override experimental data with user inputs
    data['Humid_air'][0] = T_gin
    data['Cooling_water'][0] = T_cout
    data['steam_flowrate'] = steam_flowrate/3600
    data['Mixture_flowrate'] = Air_flowrate/3600
    
    # Hardcoded relaxation factors
    alpha_G = 0.1      # Air temperature relaxation
    alpha_W = 0.99     # Wall temperature relaxation
    alpha_C = 1     # Cooling water relaxation
    alpha_cond = 0.3   # Condensation rate relaxation
    
    # Initialize result containers
    results = {
        'y_H2o': [], 'Sat_temp': [], 'Water_density': [], 'Mass_flowrate': [],
        'Water_velocity': [], 'Water_Dynamic_viscosity': [], 'Water_Reynolds': [],
        'Latent_heat_air': [], 'Lewis_air': [], 'Temperature_interface': [], 
        'Vapour_mole_interface': [], 'Mass_transfer_coefficient_air': [], 
        'Logarithmic_mole_average': [], 'Inlet_temp_air': [], 'Outlet_temp_air': [], 
        'Inlet_temp_water': [], 'Condensation_rate': [], 'Outlet_temp_water': [],
        'Wall_temperature2': []
    }
    
    m_frac = data['steam_flowrate'] / data['Mixture_flowrate']
    M_frac = m_frac
    Condensation_rate_total = 0
    
    # Initialize temperatures for the first iteration
    T_cin = data['Cooling_water'][0]
    T_gin = data['Humid_air'][0]
    cond_error_history = []  # To track convergence behavior
    prev_cond_rate = None  # To store previous condensation rate

    for i in range(n):
        # Current segment calculations
        T_cout = T_cin  # Outlet becomes inlet for next segment
        
        # Water properties
        T_c_avg = (T_cin + T_cout)/2 if i > 0 else T_cout
        rho_c, k_c, c_pc, u_c = calculate_water_properties(T_c_avg)
        m_c = data['CW_flowrate'] * rho_c / (3600*1000*3)
        A_c = (((math.pi * D_i**2) / 4 )*8)/n 
        v_c = m_c / (rho_c * A_c)
        Re_c = (rho_c * v_c * D_i) / u_c
        alpha_c = k_c / (rho_c * c_pc)
        pr_c = (u_c/rho_c) / alpha_c
        Nu_c, h_c = calculate_nusselt_water(Re_c, pr_c, k_c, D_i, x_values, M_Nu)
        
        # Wall temperature estimation
        if i == 0:
            T_w = (a/2)*(T_gin + T_cout)
        else:
            delta_Ai = ((0.364 * math.pi * D_i)*8)/n
            T_w_calc = (m_c * c_pc * (T_cout - T_cin)) / (h_c * delta_Ai)
            T_w = alpha_W * results['Wall_temperature2'][-1] + (1 - alpha_W) * T_w_calc
        
        # Air properties
        y_h2o = (float(M_frac)/M_h2o) / ((float(M_frac)/M_h2o) + ((1 - float(M_frac))/M_g))
        P_w = y_h2o * 1
        T_sat = np.interp(P_w, pressure, temperature)
        
        u_g, k_g, c_pg = calculate_air_properties(T_gin, M_frac)
        rho_air = ((p*M_g/1000)/(R_air*(T_gin+273))) * (1-y_h2o)
        rho_water = ((p*M_h2o/1000)/(R_water*(T_gin+273))) * y_h2o
        rho_g = rho_air + rho_water
        
        m_g = data['Mixture_flowrate'] - Condensation_rate_total
        A_gap = (0.011*8)/n
        v_g = m_g / (rho_g * A_gap)
        Re_g = (rho_g * v_g * D_o) / u_g
        alpha_g = k_g / (rho_g * c_pg)
        pr_g = (u_g/rho_g) / alpha_g
        Nu_g = calculate_nusselt_air(Re_g, pr_g, k_g)
        h_g = (Nu_g * k_g) / D_o
        
        # Latent heat
        h_fg = PropsSI("H", "T", T_w+273.1, "Q", 1, "Water") - PropsSI("H", "T", T_w+273.1, "Q", 0, "Water")
        D_h2oair = (6.057e-6 + 4.055e-8*(T_gin+273) + 1.25e-10*(T_gin+273)**2 - 3.367e-14*(T_gin+273)**3)
        Le_h20air = alpha_g / D_h2oair
        
        # Interface calculation
        if T_w < T_sat:
            def equation(T_i):
                y_i = np.exp(a_antoine - (b_antoine/(T_i + c_antoine))) / (p/1000)
                y_ni = 1 - y_i
                y_nb = 1 - y_h2o
                y_lm = (y_ni - y_nb) / math.log(y_ni/y_nb) if y_ni != y_nb else y_nb
                k_m = (h_g * M_h2o) / (c_pg * M_g * y_lm * Le_h20air**(2/3))
                return ((h_g*T_gin + h_fg*k_m*(y_h2o-y_i) + h_c*T_cout)/(h_g + h_c) - T_i)
            
            try:
                T_i_solution = newton(equation, 70, maxiter=100)
                y_i = np.exp(a_antoine - (b_antoine/(T_i_solution + c_antoine))) / (p/1000)
                y_lm = (1-y_i - (1-y_h2o)) / math.log((1-y_i)/(1-y_h2o)) if (1-y_i) != (1-y_h2o) else (1-y_h2o)
                k_m = (h_g * M_h2o) / (c_pg * M_g * y_lm * Le_h20air**(2/3))
                
                delta_Ai = (0.0206*8)/n
                if i == 0:
                    T_gout = ((m_g*c_pg - (h_g/2)*delta_Ai)*T_gin + h_g*delta_Ai*T_i_solution) / \
                            (m_g*c_pg + (h_g/2)*delta_Ai)
                    T_cin_new = T_cout - ((h_g*(T_gin-T_i_solution)*delta_Ai + h_fg*k_m*(y_h2o-y_i)*delta_Ai)/(m_c*c_pc))
                    m_cd = k_m * (y_h2o - y_i) * delta_Ai
                else:
                    T_gout_calc = ((m_g*c_pg - (h_g/2)*delta_Ai)*T_gin + h_g*delta_Ai*T_i_solution) / \
                            (m_g*c_pg + (h_g/2)*delta_Ai)
                    T_gout = alpha_G * results['Outlet_temp_air'][-1] + (1 - alpha_G) * T_gout_calc
                    
                    T_cin_calc = T_cout - ((h_g*(T_gin-T_i_solution)*delta_Ai + h_fg*k_m*(y_h2o-y_i)*delta_Ai)/(m_c*c_pc))
                    T_cin_new = alpha_C * results['Inlet_temp_water'][-1] + (1 - alpha_C) * T_cin_calc
                    
                    m_cd_calc = k_m * (y_h2o - y_i) * delta_Ai
                    current_error_cond = abs(m_cd_calc - results['Condensation_rate'][-1])
                    cond_error_history.append(current_error_cond)
                    # Adaptive adjustment (only after we have some history)
                    if len(cond_error_history) > 1:
                        if current_error_cond < cond_error_history[-2]:  # Converging
                            alpha_cond = min(alpha_cond +0.02, 0.5)  # Increase relaxation
                            print("Converging")
                        else:  # Diverging
                            alpha_cond = max(alpha_cond +0.05, 0.01)  # Decrease relaxation
                            print("Diverging")
                    m_cd = results['Condensation_rate'][-1] + (alpha_cond * (m_cd_calc - results['Condensation_rate'][-1]))
                
                Condensation_rate_total += m_cd
                M_frac = (data['steam_flowrate'] - Condensation_rate_total) / \
                        (data['Mixture_flowrate'] - Condensation_rate_total)
                
            except RuntimeError:
                T_gout = T_gin
                T_cin_new = T_cout
                m_cd = 0
        else:
            delta_Ai = (0.0206*8)/n
            if i == 0:
                T_gout = ((m_g*c_pg - (h_g/2)*delta_Ai)*T_gin + h_g*delta_Ai*T_w) / \
                        (m_g*c_pg + (h_g/2)*delta_Ai)
                T_cin_new = T_cout - ((h_g*(T_gin-T_w)*delta_Ai)/(m_c*c_pc))
            else:
                T_gout_calc = ((m_g*c_pg - (h_g/2)*delta_Ai)*T_gin + h_g*delta_Ai*T_w) / \
                        (m_g*c_pg + (h_g/2)*delta_Ai)
                T_gout = alpha_G * results['Outlet_temp_air'][-1] + (1 - alpha_G) * T_gout_calc
                T_cin_calc = T_cout - ((h_g*(T_gin-T_w)*delta_Ai)/(m_c*c_pc))
                T_cin_new = alpha_C * results['Inlet_temp_water'][-1] + (1 - alpha_C) * T_cin_calc
            m_cd = 0
        
        # Update for next iteration
        T_gin = T_gout
        T_cin = T_cin_new
        
        # Store results
        results['y_H2o'].append(y_h2o)
        results['Sat_temp'].append(T_sat)
        results['Water_density'].append(rho_c)
        results['Mass_flowrate'].append(m_c)
        results['Water_velocity'].append(v_c)
        results['Water_Dynamic_viscosity'].append(u_c)
        results['Water_Reynolds'].append(Re_c)
        results['Latent_heat_air'].append(h_fg)
        results['Lewis_air'].append(Le_h20air)
        results['Inlet_temp_air'].append(T_gin)
        results['Outlet_temp_air'].append(T_gout)
        results['Inlet_temp_water'].append(T_cin_new)
        results['Condensation_rate'].append(m_cd)
        results['Outlet_temp_water'].append(T_cout)
        results['Wall_temperature2'].append(T_w)
        
        if T_w < T_sat and 'T_i_solution' in locals():
            results['Temperature_interface'].append(T_i_solution)
            results['Vapour_mole_interface'].append(y_i)
            results['Mass_transfer_coefficient_air'].append(k_m)
            results['Logarithmic_mole_average'].append(y_lm)
        else:
            results['Temperature_interface'].append(np.nan)
            results['Vapour_mole_interface'].append(np.nan)
            results['Mass_transfer_coefficient_air'].append(np.nan)
            results['Logarithmic_mole_average'].append(np.nan)
    
    return results, data

def main():
    st.title("Condensing Heat Exchanger Simulator")
    
    # Load experimental data first
    df = load_data()
    
    # Sidebar controls with experimental values shown
    st.sidebar.header("Simulation Parameters")
    
    # Experiment number selector
    e = st.sidebar.slider("Experiment number", 0, 100, 53, 1)
    data = get_experiment_data(df, e)
    
    # Display experimental values in the main area
    st.subheader(f"Experimental Values for Experiment {e}")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Humid Air Inlet Temp (°C)", 
                 f"{data['Humid_air'][0]:.1f}")
        
        st.metric("Cooling Water Outlet Temp (°C)", 
                 f"{data['Cooling_water'][0]:.1f}")
        
    with col2:
        st.metric("Vapour Flow Rate (kg/h)", 
                 f"{data['steam_flowrate']*3600:.2f}")
        
        st.metric("Cooling Water Flow Rate (l/h)", 
                 f"{data['CW_flowrate']:.1f}",
                 help="Experimental cooling water flow rate")
        
    with col3:
        st.metric("Air Flow Rate (kg/h)", 
                 f"{data['Mixture_flowrate']*3600:.1f}")
        
        st.metric("Condensation Rate (kg/h)", 
                 f"{data['exp_cond_rate']*3600:.3f}")
    
    # Sliders with experimental values as default values
    st.sidebar.subheader("Simulation Inputs")
    
    # Get experimental values to use as defaults
    exp_T_gin = data['Humid_air'][0]
    exp_T_cout = data['Cooling_water'][0]
    exp_steam = data['steam_flowrate']*3600
    exp_air = data['Mixture_flowrate']*3600
    exp_cw_flow = data['CW_flowrate']
    
    # Create sliders with experimental defaults
    T_gin = st.sidebar.slider(
        'Humid air inlet temperature (°C)',
        0, 250, int(exp_T_gin), 1,
        help=f"Experimental value: {exp_T_gin:.1f}°C"
    )
    
    T_cout = st.sidebar.slider(
        'Cooling water outlet temperature (°C)',
        0, 100, int(exp_T_cout), 1,
        help=f"Experimental value: {exp_T_cout:.1f}°C"
    )
    
    steam_flowrate = st.sidebar.slider(
        'Vapour flow rate (kg/h)',
        0, 200, int(exp_steam), 1,
        help=f"Experimental value: {exp_steam:.2f} kg/h"
    )
    
    Air_flowrate = st.sidebar.slider(
        'Air flow rate (kg/h)',
        0, 2000, int(exp_air), 1,
        help=f"Experimental value: {exp_air:.1f} kg/h"
    )
    
    CW_flowrate = st.sidebar.slider(
        'Cooling water flow rate (l/h)',
        0, 3000, int(exp_cw_flow), 1,
        help=f"Experimental value: {exp_cw_flow:.1f} l/h"
    )
    
    a = st.sidebar.slider(
        'Wall temperature coefficient',
        0.0, 1.0, 0.65, 0.01
    )
    
    n = st.sidebar.slider(
        "Number of iterations",
        1, 50, 40, 1
    )
    
    # Hardcoded relaxation factors (removed from UI)
    st.sidebar.header("Relaxation Factors (Hardcoded)")
    st.sidebar.text("Air temp relaxation: 0.1")
    st.sidebar.text("Wall temp relaxation: 0.99")
    st.sidebar.text("Cooling water relaxation: 0.75")
    st.sidebar.text("Condensation rate relaxation: 0.1")
    
    # Hardcoded relaxation factors (removed from UI)
    # st.sidebar.header("Relaxation Factors (Hardcoded)")
    # st.sidebar.text("Air temp relaxation: 0.1")
    # st.sidebar.text("Wall temp relaxation: 0.99")
    # st.sidebar.text("Cooling water relaxation: 0.75")
    # st.sidebar.text("Condensation rate relaxation: 0.1")
    
    # Run simulation automatically when parameters change
    try:
        with st.spinner('Calculating...'):
            results, data = run_simulation(e, T_gin, T_cout, steam_flowrate, Air_flowrate, a, n)
        
        # Display results
        # Display results
        st.subheader("Temperature Profiles")
        
        # Get experimental data for plotting
        HA_exp = data['Humid_air']  # Experimental humid air temperatures
        CW_exp = data['Cooling_water']  # Experimental cooling water temperatures
        WT_exp = data['Wall_temp']  # Experimental wall temperatures
        
        # Create x-axis points for experimental data
        segments = list(range(1, n+1))
        exp_points_ha = np.linspace(1, n, len(HA_exp))
        exp_points_cw = np.linspace(1, n, len(CW_exp))
        exp_points_wt = np.linspace(1, n, len(WT_exp))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add traces - EXPERIMENTAL DATA (scatter only)
        fig.add_trace(go.Scatter(
            x=exp_points_ha, y=HA_exp,
            mode='markers',
            name='Experimental HA. T',
            marker=dict(color='red', size=8, symbol='circle'),
            line=dict(width=0)  # No line
        ))
        
        fig.add_trace(go.Scatter(
            x=exp_points_cw, y=CW_exp,
            mode='markers',
            name='Experimental CW. T',
            marker=dict(color='blue', size=8, symbol='square'),
            line=dict(width=0)  # No line
        ))
        
        fig.add_trace(go.Scatter(
            x=exp_points_wt, y=WT_exp,
            mode='markers',
            name='Experimental WT',
            marker=dict(color='black', size=8, symbol='diamond'),
            line=dict(width=0)  # No line
        ))
        
        # Add traces - CALCULATED DATA (dashed lines only)
        fig.add_trace(go.Scatter(
            x=segments, y=results['Outlet_temp_air'],
            mode='lines',
            name='Calculated HA. T',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=0)  # No markers
        ))
        
        fig.add_trace(go.Scatter(
            x=segments, y=results['Inlet_temp_water'],
            mode='lines',
            name='Calculated CW. T',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(size=0)  # No markers
        ))
        
        fig.add_trace(go.Scatter(
            x=segments, y=results['Wall_temperature2'],
            mode='lines',
            name='Calculated WT',
            line=dict(color='black', width=2, dash='dash'),
            marker=dict(size=0)  # No markers
        ))
        
        # Update layout
        fig.update_layout(
            title='Temperature Profiles Along the Heat Exchanger',
            xaxis_title='Segment Number',
            yaxis_title='Temperature (°C)',
            legend_title='Temperature Type',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        # Customize legend to group experimental/calculated
        fig.update_layout(
            legend=dict(
                traceorder="grouped",
                itemsizing='constant'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Condensation results
        total_condensation = sum(results['Condensation_rate'])*3600  # Convert to kg/h
        exp_cond = data['exp_cond_rate']*3600 
        
        st.subheader("Condensation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Experimental Condensation", f"{exp_cond:.2f} kg/h")
            st.metric("Calculated Condensation", f"{total_condensation:.2f} kg/h")
        with col2:
            st.metric("Difference", f"{abs(exp_cond-total_condensation):.2f} kg/h")
            if exp_cond > 0:
                st.metric("Relative Error", f"{abs(exp_cond-total_condensation)/exp_cond*100:.1f}%")
        
        # Detailed results
        if st.checkbox("Show detailed results"):
            st.dataframe(pd.DataFrame({
                'Iteration': segments,
                'Air_Inlet_Temp': results['Inlet_temp_air'],
                'Air_Outlet_Temp': results['Outlet_temp_air'],
                'Water_Inlet_Temp': results['Inlet_temp_water'],
                'Water_Outlet_Temp': results['Outlet_temp_water'],
                'Wall_Temp': results['Wall_temperature2'],
                'Condensation_Rate_kg/s': results['Condensation_rate'],
                'Interface_Temp': results['Temperature_interface'],
                'Vapor_Mole_Fraction': results['y_H2o']
            }))
            
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")

if __name__ == "__main__":
    main()
