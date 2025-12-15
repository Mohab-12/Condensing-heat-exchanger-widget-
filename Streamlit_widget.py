import numpy as np
import pandas as pd
import math
from scipy.optimize import newton
import warnings
warnings.simplefilter("ignore")
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
import matplotlib.pyplot as plt
import streamlit as st
# %matplotlib inline

# --- Configuration / constants -------------------------------------------------
file_path = r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx"
M_h2o = 18.015  # g/mol (unused for dimensionless ratios but kept)
M_g = 28.96
L = 0.364
total_length_per_coil = 2.85  # from geometry table
tube_n = 8 # number of vertical  tubes in one serepentine coil
coil_n = 3 # number of serepentune coils in the heat exchanger
Ao_total_measured = 0.49375872      # ALL serpentines outer area [m²]
D_i = 0.014 # Inner diameter
D_o = 0.018 # Outer diameter
p = 101325
R_air = 8.314
R_water = 8.314
# Antoine-ish coefficients (kept from original)
a_antoine = 16.262
b_antoine = 3799.89
c_antoine = 226.35
pressure = [0.15, 0.2, 0.25, 0.5]
temperature = [53.983, 60.073, 64.980, 81.339]

# geometry / lookup
x_values = [0.001, 0.004, 0.01, 0.04, 0.08, 0.1, 0.2]
M_Nu = [19.29, 12.09, 8.92, 5.81, 4.86, 4.64, 4.15]

# --- Helpers ------------------------------------------------------------------
def load_data(path=file_path):
    df = pd.read_excel(path, sheet_name='Sheet3')
    df['Date'] = df['Date'].astype(str)
    return df

def get_experiment_data(df, e):
    return {
        'Humid_air': df.loc[e, [f'Temperature decrease of the mixture_{i}' for i in range(1,10)]].values,
        'Cooling_water': df.loc[e, [f'Temperature increase of the cooling water_{i}' for i in range(1,10)]].values,
        'Wall_temp': df.loc[e, [f'Tube_coil_surface_temp{i}' for i in range(1,9)]].values,
        'dew_point': df.loc[e, [f'Dew point_{i}' for i in range(1,9)]].values,
        'steam_flowrate': df.loc[e, 'Vapour flow rate, kg/h'] / 3600.0,      # kg/s
        'CW_flowrate': df.loc[e, 'Cooling water flow rate, l/h'],            # L/h
        'Mixture_flowrate': df.loc[e, 'Mixture  (air+vapour) flow rate, kg/h'] / 3600.0,  # kg/s
        'Condensate_flow_rate': df.loc[e,'Condensation flow rate(Kg/min)']
    }

def water_props_from_temp_C(T_c):
    T_k = max(T_c + 273.15, 273.2)
    rho = PropsSI('D', 'T', T_k, 'P', p, 'Water') #    # Density of water at given temperature [kg/m³]
    k = PropsSI('CONDUCTIVITY', 'T', T_k, 'P', p, 'Water')    # Thermal conductivity of water at given temperature [W/m/K]
    cp = PropsSI('C', 'T', T_k, 'P', p, 'Water') #    Specific heat capacity at constant pressure [J/kg/K]
    mu = PropsSI('V', 'T', T_k, 'P', p, 'Water')    # Dynamic viscosity [Pa·s = kg/(m·s)]
    return rho, k, cp, mu

def air_props_from_T_and_Mfrac(T_g_C, M_frac_mass):
    # CoolProp expects humidity ratio W = kg_vapor / kg_dry_air
    W = M_frac_mass / (1.0 - M_frac_mass)
    T_k = T_g_C + 273.15
    mu = HAPropsSI('mu', 'T', T_k, 'P', p, 'W', W)     # Dynamic viscosity of humid air [Pa·s]
    k = HAPropsSI('k', 'T', T_k, 'P', p, 'W', W)    # Thermal conductivity of humid air [W/m/K]
    cha = HAPropsSI('Cha', 'T', T_k, 'P', p, 'W', W)      # Specific heat capacity of humid air [J/kg/K]
    return mu, k, cha

def nusselt_water(Re_c, pr_c, k_c, D_i, n_segments):
    if Re_c < 3000:
        total_length_per_coil = 2.85  # from geometry table
        delta_L = total_length_per_coil / n_segments
        x = delta_L
        x_ = ((2 * x) / D_i) / (Re_c * pr_c)
        Mean_Nu = np.interp(x_, x_values, M_Nu)
        h = Mean_Nu * k_c / D_i
        return Mean_Nu, h
    else:
        if Re_c <= 2e4:
            f = 0.316 * Re_c ** -0.25
        else:
            f = (0.790 * math.log(Re_c) - 1.64) ** -2
        Nu_c = (f / 8) * (Re_c - 1000) * pr_c / (1 + 12.7 * math.sqrt(f / 8) * (pr_c ** (2 / 3) - 1))
        h = Nu_c * k_c / D_i
        return Nu_c, h

def nusselt_air(Re_g, pr_g, k_g):
    if Re_g < 1e3 or Re_g > 2e6:
        # out of nominal range; you can still use it, but be aware
        pass
    return 0.27 * (Re_g**0.63) * (pr_g**0.36)

# --- Main: corrected segment loop --------------------------------------------
def run_segmental_model(e, n_segments=40, debug=False):
    df = load_data()
    data = get_experiment_data(df, e)

    tolerance = 3.0
    max_iterations = 100
    iteration = 0

    Ao_total_measured = 0.49375872      # ALL serpentines outer area [m²]
    Ao_per_serpentines = Ao_total_measured / coil_n  # 0.16458624 m² per serpentine
    delta_Ao = Ao_per_serpentines / n_segments # total area per segment m²
    
    Tg = np.zeros(n_segments + 1)
    Tc = np.zeros(n_segments + 1)   # Tc[i] is water temperature at junction i between segments
    Tw = np.zeros(n_segments + 1)      # wall temperature per segment (or film approximation)
    m_cd = np.zeros(n_segments) # condensation per segment
    
    Tc_target = float(data['Cooling_water'][-1])
    T_c_guess = Tc[0] = float(data['Cooling_water'][0])
    
    while iteration < max_iterations:
        # results container
        results = {k: [] for k in [
            'y_H2o','Sat_temp','Water_density','Mass_flowrate','Water_velocity',
            'Water_Dynamic_viscosity','Water_Reynolds','Water_thermal_conductivity',
            'Water_Nusselt_number','Water_heat_Transfer_coefficient','Water_specific_heat',
            'Wall_temperature2','Density_air','FlowRate_air','Velocity_air','Specific_heat_air',
            'Viscosity_air','Reynolds_air','Prandtl','Thermal_conductivity_air','Thermal_diffusivity_air',
            'Nusselt_air','Heat_transfer_air','Latent_heat_air','Lewis_air','Mass_of_diffusivity',
            'Outlet_temp_air','Inlet_temp_water','Condensation_rate','Imbalance',
            'Q_Air_Sensible','Cond','Water','HTA','HTW'
        ]}
        
        Tg = np.zeros(n_segments + 1)
        Tc = np.zeros(n_segments + 1)   # Tc[i] is water temperature at junction i between segments
        Tw = np.zeros(n_segments + 1)      # wall temperature per segment (or film approximation)
        m_cd = np.zeros(n_segments) # condensation per segment
        # Condensation_rate_total = 0.0
        M_frac = data['steam_flowrate'] / data['Mixture_flowrate']
        Tg_inlet = Tg[0] = float(data['Humid_air'][0])
        Tc[0] = T_c_guess
        T_w = Tw[0] = float(data['Wall_temp'][0])
        
        for seg in range(n_segments):
            # Assignning temperatures
            T_g_inlet = Tg[seg] 
            T_c_outlet = Tc[seg]
            # --- water side (using inlet estimate) ---------------------------------
            # use inlet temperature as start; after computing Q we get outlet
            rho_c, k_c, c_pc, mu_c = water_props_from_temp_C(Tc[seg])
            # convert cooling water flow: L/h -> m^3/s
            mdot_total_cw_m3s = data['CW_flowrate'] / 3600.0 * 1e-3
            # assumption: parallel 3 branches (user) -> per-branch volumetric flow
            mdot_branch_m3s = mdot_total_cw_m3s / 3.0
            m_c = mdot_branch_m3s * rho_c  # kg/s per branch
    
            A_c = (math.pi * D_i**2 / 4.0)  # Corss sectional area of the serepentine
            v_c = m_c / (rho_c * A_c + 1e-12)
            Re_c = (rho_c * v_c * D_i) / (mu_c + 1e-12)
            alpha_c = k_c / (rho_c * c_pc)
            pr_c = (mu_c / rho_c) / alpha_c
            Nu_c, h_c = nusselt_water(Re_c, pr_c, k_c, D_i, n_segments)
            # print(f"Seg{seg:2d}: h_c={h_c:5.0f} Re_c={Re_c:5.0f} m_c={m_c*3600:5.1f}kg/h Tw={Tw[seg]:5.1f}")

    
            # --- gas side properties (use current M_frac and T_g_inlet) ------------
            # mass fraction M_frac = m_vapor / (m_vapor + m_dry) - 
            # Convert to humidity ratio W = m_vapor/m_dry for CoolProp
            mu_g, k_g, c_pg = air_props_from_T_and_Mfrac(Tg[seg], M_frac)
    
            # compute partial densities (approx.) and mixture density
            y_h2o = (M_frac / M_h2o) / ((M_frac / M_h2o) + ((1.0 - M_frac) / M_g))
            rho_air = ((p * M_g / 1000.0) / (R_air * (Tg[seg] + 273.15))) * (1.0 - y_h2o)
            rho_vap = ((p * M_h2o / 1000.0) / (R_water * (Tg[seg] + 273.15))) * y_h2o
            rho_g = rho_air + rho_vap
    
            # current mixture mass flow available downstream (subtract what already condensed)
            m_g = max(data['Mixture_flowrate'] - (coil_n * np.sum(m_cd)), 1e-12)
            A_gap = 0.011 # Area where the humid air flows >>> Cross sectional area of Rectangular duct - Cross section area of coils
            v_g = m_g / (rho_g * A_gap + 1e-12)
            Re_g = (rho_g * v_g * D_o) / (mu_g + 1e-12)
            alpha_g = k_g / (rho_g * c_pg + 1e-12)
            pr_g = (mu_g / rho_g) / (alpha_g + 1e-12)
            Nu_g = nusselt_air(Re_g, pr_g, k_g)
            h_g = (Nu_g * k_g) / D_o
    
            # vapour diffusivity estimate and Lewis number
            D_h2oair = (6.057e-6 + 4.055e-8 * (Tg[seg] + 273.15) + 1.25e-10 * (Tg[seg] + 273.15) ** 2 - 3.367e-14 * (Tg[seg] + 273.15) ** 3)
            Le_h2oair = alpha_g / (D_h2oair + 1e-20)
    
            # --- interface and condensation calculation ---------------------------
            if Tw[seg] < HAPropsSI('Tdp','T',Tg[seg]+273.15,'P',p,'W',M_frac / (1.0 - M_frac)):
                # Solve interface temperature T_i (user's original equation preserved)
                def equation_Ti(T_i):
                    y_i = np.exp(a_antoine - (b_antoine / (T_i + c_antoine))) / (p / 1000.0)
                    y_ni = 1.0 - y_i
                    y_nb = 1.0 - y_h2o
                    y_lm = (y_ni - y_nb) / math.log(y_ni / y_nb) if (y_ni != y_nb) else y_nb
                    k_m = (h_g * M_h2o) / (c_pg * M_g * y_lm * (Le_h2oair ** (2.0 / 3.0)) + 1e-20)
                    h_fg = PropsSI('H','T',(Tw[seg] if seg == 0 else T_i) + 273.15,'Q',1,'Water') \
                  - PropsSI('H','T',(Tw[seg] if seg == 0  else T_i) + 273.15,'Q',0,'Water')
                    return (h_g * Tg[seg] + h_fg * k_m * (y_h2o - y_i) + h_c * Tc[seg]) / (h_g + h_c + 1e-20) - T_i
                try:
                    T_i = newton(equation_Ti, 60.0, maxiter=100)
                    y_i = np.exp(a_antoine - (b_antoine / (T_i + c_antoine))) / (p / 1000.0)
                    y_lm = (1.0 - y_i - (1.0 - y_h2o)) / math.log((1.0 - y_i) / (1.0 - y_h2o)) if (1.0 - y_i) != (1.0 - y_h2o) else (1.0 - y_h2o)
                    # # latent heat at approximate wall temperature guess
                    # approximate wall temperature from previous energy balance (user's approach)
                    h_fg = PropsSI('H','T',(Tw[seg] if seg == 0 else T_i) + 273.15,'Q',1,'Water') \
                           - PropsSI('H','T',(Tw[seg] if seg == 0  else T_i) + 273.15,'Q',0,'Water')
                    k_m = (h_g * M_h2o) / (c_pg * M_g * y_lm * (Le_h2oair ** (2.0 / 3.0)) + 1e-20)
    
                    # temperature changes across this segment (energy balance style formulas from original)
                    Tg[seg+1] = ((m_g * c_pg - (h_g / 2.0) * delta_Ao* coil_n) * Tg[seg] + h_g * delta_Ao* coil_n * T_i) / (m_g * c_pg + (h_g / 2.0) * delta_Ao* coil_n + 1e-20)
                    Tc[seg+1] = Tc[seg] - ((h_g * (Tg[seg] - T_i) * delta_Ao) + h_fg * k_m * (y_h2o - y_i) * delta_Ao) / (m_c * c_pc + 1e-20)
    
                    # local condensation mass flux (per this segment / per branch)
                    m_cd[seg] = k_m * (y_h2o - y_i) * delta_Ao
                    m_cd_total = coil_n * np.sum(m_cd)

    
                    # update totals and compositions
                    M_frac = max((data['steam_flowrate'] - m_cd_total) / max((data['Mixture_flowrate'] - m_cd_total), 1e-12), 0.0)
    
                except RuntimeError:
                    # fallback: no condensation
                    T_i = 0.0
                    m_cd[seg] = 0.0
                    h_fg = 0.0
            else:
                # no condensation: sensible heat only
                Tg[seg+1] = ((m_g * c_pg - (h_g / 2.0) * delta_Ao* coil_n) * Tg[seg] + h_g * delta_Ao* coil_n * Tw[seg]) / (m_g * c_pg + (h_g / 2.0) * delta_Ao* coil_n + 1e-20)
                Tc[seg+1] = Tc[seg] - ((h_g * (Tg[seg] - Tw[seg]) * delta_Ao) / (m_c * c_pc + 1e-20))
                m_cd[seg] = 0.0
                h_fg = 0.0
    
            # Tw[seg+1] = ((Tc[seg] + Tc[seg+1])/2) + ((m_c * c_pc * (Tc[seg] - Tc[seg+1])) / (h_c * delta_Ao + 1e-12))
            # Tw[seg+1] = Tc[seg] + ((m_c * c_pc * (Tc[seg] - Tc[seg+1])) / (h_c * delta_Ao + 1e-12))
            Aii = np.pi * D_i * (total_length_per_coil / n_segments)  # per serpentine
            Aii = np.pi * D_i * (total_length_per_coil / n_segments)  # per serpentine            
            Tw[seg+1] = (Tc[seg] + Tc[seg+1])/2 + (m_c * c_pc * (Tc[seg] - Tc[seg+1])) / (2 * h_c * Aii)
            # print("T_wall", Tw[seg+1])
            # print("Water inlet", Tc[seg+1])
            # energy bookkeeping
            # Gas total heat (already in Tg update)
            Q_gas_total = m_g * c_pg * (Tg[seg] - Tg[seg+1])
            
            # Per coil for water comparison
            Q_gas_per_coil = Q_gas_total / coil_n
            Q_cond_per_coil = m_cd[seg] * h_fg
            Q_water_per_coil = m_c * c_pc * (Tc[seg] - Tc[seg+1])
            imbalance = abs((Q_gas_per_coil + Q_cond_per_coil) - Q_water_per_coil)

            # print(f"Seg{seg:2d}: h_c={h_c:4.0f} h_g={h_g:4.0f} Re_g={Re_g:6.0f} Tg_out={Tg[-1]:4.1f} Tw={Tw[seg+1]:4.1f}")
            # print(f"Total cond: {coil_n*np.sum(m_cd)*3600:.1f} kg/h vs measured {data['Condensate_flow_rate']*60:.1f}")
            
            # save results for this segment
            results['y_H2o'].append(y_h2o)
            results['Sat_temp'].append(np.interp(y_h2o, pressure, temperature))
            results['Water_density'].append(rho_c)
            results['Mass_flowrate'].append(m_c)
            results['Water_velocity'].append(v_c)
            results['Water_Dynamic_viscosity'].append(mu_c)
            results['Water_Reynolds'].append(Re_c)
            results['Water_thermal_conductivity'].append(k_c)
            results['Water_Nusselt_number'].append(Nu_c)
            results['Water_heat_Transfer_coefficient'].append(h_c)
            results['Water_specific_heat'].append(c_pc)
            results['Wall_temperature2'].append(Tw[seg+1])
            results['Density_air'].append(rho_g)
            results['FlowRate_air'].append(m_g)
            results['Velocity_air'].append(v_g)
            results['Specific_heat_air'].append(c_pg)
            results['Viscosity_air'].append(mu_g)
            results['Reynolds_air'].append(Re_g)
            results['Prandtl'].append(pr_g)
            results['Thermal_conductivity_air'].append(k_g)
            results['Thermal_diffusivity_air'].append(alpha_g)
            results['Nusselt_air'].append(Nu_g)
            results['Heat_transfer_air'].append(h_g)
            results['Latent_heat_air'].append(h_fg)
            results['Lewis_air'].append(Le_h2oair)
            results['Mass_of_diffusivity'].append(D_h2oair)
            results['Outlet_temp_air'].append(Tg[seg+1])
            results['Inlet_temp_water'].append(Tc[seg+1])
            results['Condensation_rate'].append(m_cd[seg])
            results['Imbalance'].append(imbalance)
            results['Q_Air_Sensible'].append(Q_gas_per_coil)
            results['Cond'].append(Q_cond_per_coil)
            results['Water'].append(Q_water_per_coil)
            results['HTA'].append(h_g)
            results['HTW'].append(h_c)

            
        iteration= iteration+1      
        error = Tc[-1] - Tc_target
        if abs(error) < tolerance:
            print("Water segment temps:", T_c_guess)
            break
        # If not converged update the guess
        elif error<=0:
            # print("Water segment temps:", T_c_guess)
            T_c_guess = T_c_guess + 1
        elif error>tolerance:
            # print("Water segment temps:", T_c_guess)
            T_c_guess = T_c_guess - 1
                    
    return results, iteration, data

if __name__ == '__main__':
    # Example: run experiment 60
    res, iteration, exp_data = run_segmental_model(60, n_segments=100, debug=True)
    # print short summary
    print('Done. Segments:', len(res['Outlet_temp_air']))
    print('Number of iterations are: ', iteration)
# --- Streamlit Dashboard ---------------------------------------------------

st.set_page_config(page_title="Heat Exchanger Dashboard", layout="wide")

st.title("🌡️ Condensing Heat Exchanger – Streamlit Dashboard")
st.write("Interactive exploration of your segmental heat-exchanger model.")

# --- File Upload -----------------------------------------------------------

with st.sidebar:
    st.header("📂 Load Experimental Data")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded:
        df = load_data(uploaded)
    else:
        st.info("Using default file.")
        df = load_data(file_path)

    exp_id = st.number_input("Experiment ID", min_value=0, max_value=len(df)-1, value=60)
    # n_segments = st.slider("Number of segments", 20, 200, 40)
    n_segments = 40
    run_button = st.button("Run Model")

# --- Run model -------------------------------------------------------------

if run_button:
    with st.spinner("Running segmental model..."):
        results, iterations, exp_data = run_segmental_model(exp_id, n_segments=n_segments)

    st.success(f"Model completed in **{iterations} iterations**.")

    # Convert results to DataFrame for convenience
    # SAFE VISUALIZATION FIX (DO NOT TOUCH MODEL STRUCTURE)
    # ----------------------------------------------------------
    lengths = {k: len(v) for k, v in results.items()}
    min_len = min(lengths.values())
    
    # Optional: show warning in Streamlit if truncation happens
    if len(set(lengths.values())) > 1:
        st.warning(
            f"Result arrays had inconsistent lengths. "
            f"Truncating all to {min_len} segments for visualization."
        )
    
    # Truncate safely
    results_vis = {k: v[:min_len] for k, v in results.items()}
    
    # Create DataFrame ONLY from aligned data
    seg_df = pd.DataFrame(results_vis)
    # ----------------------------------------------------------
    # MAIN SUMMARY
    # ----------------------------------------------------------
    st.subheader("📊 Main Results Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Outlet Air Temperature", f"{seg_df['Outlet_temp_air'].iloc[-1]:.2f} °C")
    col2.metric("Intlet Water Temperature", f"{seg_df['Inlet_temp_water'].iloc[-1]:.2f} °C")
    col3.metric("Total Condensed Mass", f"{seg_df['Condensation_rate'].sum()*coil_n:.5f} kg/s")

    # ----------------------------------------------------------
    # TABS FOR ALL PLOTS
    # ----------------------------------------------------------
    tabs = st.tabs([
        "Temperature Profiles",
        "Heat Transfer Coefficients",
        "Condensation",
        "Flow Parameters",
        "Energy Balance",
        "Raw Output Data"
    ])

    # -------------------------------------------------------------------
    # 1. TEMPERATURE PROFILES
    # -------------------------------------------------------------------
    with tabs[0]:
        st.subheader("📈 Temperature Evolution Along the Heat Exchanger")
    
        # Experimental temperatures (REAL values)
        exp_air_temp = np.array(exp_data['Humid_air'])
        exp_cw_temp  = np.array(exp_data['Cooling_water'])
        
        # Map experimental points onto model segments
        x_exp_air = np.linspace(0, len(seg_df) - 1, len(exp_air_temp))
        x_exp_cw  = np.linspace(0, len(seg_df) - 1, len(exp_cw_temp))

    
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Outlet_temp_air'], label="Model Air Temperature", ls='--')
        ax.plot(seg_df['Inlet_temp_water'], label="Model Water Temperature", ls='--')
        # ax.plot(seg_df['Wall_temperature2'], label="Model Wall Temperature", ls='--')
        ax.scatter(x_exp_air, exp_air_temp, color='red', label="Experimental Air", zorder=5)
        ax.scatter(x_exp_cw, exp_cw_temp, color='blue', label="Experimental CW", zorder=5)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc=(1.01,0.68))
        st.pyplot(fig)

    # -------------------------------------------------------------------
    # 2. HEAT TRANSFER COEFFICIENTS
    # -------------------------------------------------------------------
    with tabs[1]:
        st.subheader("🔥 Heat Transfer Coefficients")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Heat_transfer_air'], label="Air-side h")
        ax.plot(seg_df['Water_heat_Transfer_coefficient'], label="Water-side h")
        ax.set_xlabel("Segment")
        ax.set_ylabel("h (W/m²K)")
        ax.legend()
        st.pyplot(fig)

    # -------------------------------------------------------------------
    # 3. CONDENSATION
    # -------------------------------------------------------------------
    with tabs[2]:
        st.subheader("💧 Condensation Profile")
    
        # Experimental condensation (single measured value)
        exp_condensation_total = float(exp_data['Condensate_flow_rate'])
    
        # Model: total condensation across all segments
        model_total_condensation = float(seg_df['Condensation_rate'].sum())*60*coil_n
    
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(['Experimental', 'Model'], [exp_condensation_total, model_total_condensation], color=['green','blue'])
        # ax.axhline(exp_condensation_total, color='green', linestyle='--', label="Experimental Condensation")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Condensation rate (kg/min)")
        ax.legend()
        st.pyplot(fig)
    
        st.write("### Cumulative Condensation")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Condensation_rate'].cumsum(), label="Model Cumulative")
        # ax.axhline(exp_condensation_total, color='green', linestyle='--', label="Experimental Total")
        ax.set_ylabel("Total condensed mass (kg/s)")
        ax.legend()
        st.pyplot(fig)    

    # -------------------------------------------------------------------
    # 4. FLOW PARAMETERS
    # -------------------------------------------------------------------
    with tabs[3]:
        st.subheader("🌬️ Flow & Transport Properties")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Reynolds_air'], label="Re_air")
        ax.plot(seg_df['Water_Reynolds'], label="Re_water")
        ax.set_yscale("log")
        ax.legend()
        st.pyplot(fig)

    # -------------------------------------------------------------------
    # 5. ENERGY BALANCE
    # -------------------------------------------------------------------
    with tabs[4]:
        st.subheader("⚖️ Energy Balance")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Q_Air_Sensible'], label="Q_sensible")
        ax.plot(seg_df['Cond'], label="Q_condensation")
        ax.plot(seg_df['Water'], label="Q_water")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Imbalance")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(seg_df['Imbalance'])
        ax.set_ylabel("|Q_in - Q_out|")
        st.pyplot(fig)

    # -------------------------------------------------------------------
    # 6. RAW OUTPUT
    # -------------------------------------------------------------------
    with tabs[5]:
        st.write("### Segment-wise output table")
        st.dataframe(seg_df)
        st.download_button(
            "Download results as CSV",
            seg_df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )

else:
    st.info("Upload a dataset and press **Run Model** to begin.")
