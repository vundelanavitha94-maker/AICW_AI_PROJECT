import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------------
# Step 1: Load and preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset (1).csv")
    df = df.replace(r'^\s*$', np.nan, regex=True)
    features = ["Wind_Power_MW", "Demand_MW", "Battery_SOC_MWh"]
    df = df.dropna(subset=features).fillna(0)
    return df, features

df, features = load_data()

scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df["Battery_Flow_MW"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# Step 2: Fuzzy Logic System
# -------------------------------
def build_fuzzy_system():
    voltage_f = ctrl.Antecedent(np.arange(100, 501, 1), 'Voltage')
    current_f = ctrl.Antecedent(np.arange(0, 101, 1), 'Current')
    soc_f = ctrl.Consequent(np.arange(-1, 2, 1), 'Flow')

    voltage_f['low'] = fuzz.trimf(voltage_f.universe, [100, 100, 200])
    voltage_f['medium'] = fuzz.trimf(voltage_f.universe, [150, 250, 350])
    voltage_f['high'] = fuzz.trimf(voltage_f.universe, [300, 500, 500])

    current_f['low'] = fuzz.trimf(current_f.universe, [0, 0, 30])
    current_f['medium'] = fuzz.trimf(current_f.universe, [20, 50, 80])
    current_f['high'] = fuzz.trimf(current_f.universe, [70, 100, 100])

    soc_f['discharge'] = fuzz.trimf(soc_f.universe, [-1, -1, 0])
    soc_f['idle'] = fuzz.trimf(soc_f.universe, [-0.2, 0, 0.2])
    soc_f['charge'] = fuzz.trimf(soc_f.universe, [0, 1, 1])

    rules = [
        ctrl.Rule(voltage_f['high'] & current_f['high'], soc_f['charge']),
        ctrl.Rule(voltage_f['low'] & current_f['low'], soc_f['discharge']),
        ctrl.Rule(voltage_f['medium'] & current_f['medium'], soc_f['idle'])
    ]

    battery_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(battery_ctrl)

battery_sim = build_fuzzy_system()

# -------------------------------
# Step 3: Streamlit UI
# -------------------------------
st.title("⚡ Battery Charging/Discharging + Grid Supply Dashboard")

with st.sidebar:
    st.header("Input Parameters")
    voltage_in = st.slider("Voltage (V)", 100, 500, 230)
    current_in = st.slider("Current (A)", 1, 100, 10)
    load_type_in = st.selectbox("Load Type", ["Resistive","Inductive","Capacitive"])
    temperature_in = st.slider("Temperature (°C)", 0, 50, 25)

# AI Prediction
sample = scaler.transform([[df["Wind_Power_MW"].mean(),
                            df["Demand_MW"].mean(),
                            df["Battery_SOC_MWh"].mean()]])
ai_pred = rf_model.predict(sample)[0] + (voltage_in*current_in)/1000
status_ai = "Charging" if ai_pred > 0 else "Discharging"

# Fuzzy Prediction
battery_sim.input['Voltage'] = voltage_in
battery_sim.input['Current'] = current_in

try:
    battery_sim.compute()
    fuzzy_val = battery_sim.output.get('Flow', 0)
except Exception as e:
    fuzzy_val = 0
    st.warning(f"Fuzzy system could not compute: {e}")

status_fuzzy = "Charging" if fuzzy_val > 0 else "Discharging" if fuzzy_val < 0 else "Idle"

# -------------------------------
# Step 4: Tabs for Results
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Prediction", "🌐 Fuzzy Logic", "📊 Alerts & Graphs", "🏡 Grid/Village Supply"])

with tab1:
    st.metric("Battery Status", status_ai, f"{ai_pred:.2f} MW")

with tab2:
    st.metric("Battery Status", status_fuzzy, f"{fuzzy_val:.2f}")

with tab3:
    st.subheader("🚨 Alerts Dashboard")
    soc_mean = df["Battery_SOC_MWh"].mean()

    if soc_mean < 0.5:
        st.error("Low SOC Alert (<0.5 MWh)")
    elif soc_mean > 4.5:
        st.warning("High SOC Alert (>4.5 MWh)")
    else:
        st.success("SOC within safe range")

    if temperature_in > 40:
        st.error("High Temperature Alert (>40°C)")
    elif temperature_in < 5:
        st.warning("Low Temperature Alert (<5°C)")
    else:
        st.success("Temperature Normal")

    fig, ax = plt.subplots()
    ax.plot(df["Hour"], df["Battery_SOC_MWh"], label="SOC", color="blue")
    ax.plot(df["Hour"], df["Battery_Flow_MW"], label="Flow", color="green")
    ax.axhspan(0.5, 4.5, color="yellow", alpha=0.2, label="Safe SOC Range")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MWh / MW")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab4:
    st.subheader("🏡 Grid/Village Supply")
    soc_current = df["Battery_SOC_MWh"].iloc[-1]
    surplus_power = max(0, ai_pred - (5 - soc_current))  # assume battery max 5 MWh
    if surplus_power > 0:
        st.success(f"Battery Full! Surplus {surplus_power:.2f} MW sent to Grid/Villages")
    else:
        st.info("No surplus available for external supply")
        # -------------------------------
# Step 5: Parameter Sweep Graphs
# -------------------------------
tab5 = st.tabs(["📈 Voltage Sweep", "📉 Current Sweep", "⚙️ Load Type Comparison"])

with tab5[0]:
    st.subheader("Voltage vs Battery Flow")
    voltages = np.linspace(100, 500, 50)
    ai_flows = []
    fuzzy_flows = []

    for v in voltages:
        sample = scaler.transform([[df["Wind_Power_MW"].mean(),
                                    df["Demand_MW"].mean(),
                                    df["Battery_SOC_MWh"].mean()]])
        ai_flows.append(rf_model.predict(sample)[0] + (v*current_in)/1000)

        sim = build_fuzzy_system()
        sim.input['Voltage'] = v
        sim.input['Current'] = current_in
        sim.compute()
        fuzzy_flows.append(sim.output.get('Flow', 0))

    fig_v, ax_v = plt.subplots()
    ax_v.plot(voltages, ai_flows, label="AI Prediction", color="blue")
    ax_v.plot(voltages, fuzzy_flows, label="Fuzzy Prediction", color="red")
    ax_v.set_xlabel("Voltage (V)")
    ax_v.set_ylabel("Battery Flow (MW)")
    ax_v.legend()
    ax_v.grid(True)
    st.pyplot(fig_v)

with tab5[1]:
    st.subheader("Current vs Battery Flow")
    currents = np.linspace(1, 100, 50)
    ai_flows_c = []
    fuzzy_flows_c = []

    for c in currents:
        sample = scaler.transform([[df["Wind_Power_MW"].mean(),
                                    df["Demand_MW"].mean(),
                                    df["Battery_SOC_MWh"].mean()]])
        ai_flows_c.append(rf_model.predict(sample)[0] + (voltage_in*c)/1000)

        sim = build_fuzzy_system()
        sim.input['Voltage'] = voltage_in
        sim.input['Current'] = c
        sim.compute()
        fuzzy_flows_c.append(sim.output.get('Flow', 0))

    fig_c, ax_c = plt.subplots()
    ax_c.plot(currents, ai_flows_c, label="AI Prediction", color="blue")
    ax_c.plot(currents, fuzzy_flows_c, label="Fuzzy Prediction", color="red")
    ax_c.set_xlabel("Current (A)")
    ax_c.set_ylabel("Battery Flow (MW)")
    ax_c.legend()
    ax_c.grid(True)
    st.pyplot(fig_c)

with tab5[2]:
    st.subheader("Load Type Comparison")
    load_types = ["Resistive", "Inductive", "Capacitive"]
    ai_vals = []
    fuzzy_vals = []

    for lt in load_types:
        # Simple adjustment factor for load type
        factor = 1.0 if lt == "Resistive" else 0.8 if lt == "Inductive" else 1.2
        ai_vals.append(ai_pred * factor)

        sim = build_fuzzy_system()
        sim.input['Voltage'] = voltage_in
        sim.input['Current'] = current_in
        sim.compute()
        fuzzy_vals.append(sim.output.get('Flow', 0) * factor)

    fig_l, ax_l = plt.subplots()
    x = np.arange(len(load_types))
    ax_l.bar(x - 0.2, ai_vals, width=0.4, label="AI Prediction", color="blue")
    ax_l.bar(x + 0.2, fuzzy_vals, width=0.4, label="Fuzzy Prediction", color="red")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(load_types)
    ax_l.set_ylabel("Battery Flow (MW)")
    ax_l.legend()
    st.pyplot(fig_l)


# -------------------------------
# Step 6: Degradation Analysis Tab
# -------------------------------
tab6 = st.tabs(["🧪 Degradation Analysis"])[0]

with tab6:
    st.subheader("Battery Degradation Loss & Cycle Analysis")

    # Synthetic degradation dataset based on user inputs
    timesteps = 200
    soc_values = np.random.uniform(0.2, 1.0, timesteps)
    temp_values = np.random.uniform(20, 45, timesteps)
    cycle_values = np.arange(timesteps)

    # True degradation model (capacity loss)
    true_degradation = 0.0005*cycle_values + 0.3*soc_values + 0.05*temp_values

    # Train regression model
    X_deg = np.column_stack((soc_values, temp_values))
    model = LinearRegression()
    model.fit(X_deg, true_degradation)
    y_pred = model.predict(X_deg)

    mse = mean_squared_error(true_degradation, y_pred)
    st.write(f"AI Model Mean Squared Error: {mse:.4f}")

    df_deg = pd.DataFrame({
        "Time": np.arange(timesteps),
        "SOC": soc_values,
        "Temperature": temp_values,
        "Cycles": cycle_values,
        "True_Degradation": true_degradation,
        "AI_Predicted_Degradation": y_pred
    })

    # Show degradation table and a small summary
    st.subheader("Degradation Data Table")
    st.dataframe(df_deg)

    summary = {
        'Mean Loss': df_deg['AI_Predicted_Degradation'].mean(),
        'Max Loss': df_deg['AI_Predicted_Degradation'].max(),
        'Min Loss': df_deg['AI_Predicted_Degradation'].min()
    }
    st.table(pd.DataFrame([summary]))

    # Plotting results
    figd, axs = plt.subplots(2,2, figsize=(12,8))

    axs[0,0].plot(df_deg["Time"], df_deg["SOC"], color="blue")
    axs[0,0].set_title("Battery State of Charge (SOC)")
    axs[0,0].set_xlabel("Time step"); axs[0,0].set_ylabel("SOC")

    axs[0,1].plot(df_deg["Time"], df_deg["Temperature"], color="red")
    axs[0,1].set_title("Battery Temperature")
    axs[0,1].set_xlabel("Time step"); axs[0,1].set_ylabel("°C")

    axs[1,0].plot(df_deg["Time"], df_deg["Cycles"], color="green")
    axs[1,0].set_title("Cycle Count")
    axs[1,0].set_xlabel("Time step"); axs[1,0].set_ylabel("Cycles")

    axs[1,1].plot(df_deg["Time"], df_deg["True_Degradation"], label="True", color="purple")
    axs[1,1].plot(df_deg["Time"], df_deg["AI_Predicted_Degradation"], label="AI Predicted", color="orange", linestyle="--")
    axs[1,1].set_title("Battery Degradation Prediction")
    axs[1,1].set_xlabel("Time step"); axs[1,1].set_ylabel("Capacity Loss")
    axs[1,1].legend()

    plt.tight_layout()
    st.pyplot(figd)

    # Highlight degradation status
    if y_pred[-1] > 0.8:
        st.error("⚠️ Severe Degradation: Battery capacity loss > 80%")
    elif y_pred[-1] > 0.5:
        st.warning("⚠️ Moderate Degradation: Battery capacity loss > 50%")
    else:
        st.success("✅ Battery health is within safe limits")

    # Pie chart of distribution (clearer and robust)
    try:
        _ = surplus_power
    except NameError:
        soc_current = df["Battery_SOC_MWh"].iloc[-1]
        surplus_power = max(0, ai_pred - (5 - soc_current))

    labels = ["Battery Flow (AI Pred)", "Grid/Village Supply (Surplus)"]
    ai_val = max(0.0, float(ai_pred))
    surplus_val = max(0.0, float(surplus_power))
    values = [ai_val, surplus_val]

    fig2, ax2 = plt.subplots()
    total = sum(values)
    if total <= 0:
        ax2.text(0.5, 0.5, "No positive values to display", ha='center', va='center')
    else:
        explode = (0.05, 0.05)
        def autopct(pct):
            val = pct * total / 100.0
            return f"{pct:.1f}%\n({val:.2f} MW)"
        ax2.pie(values, labels=labels, autopct=autopct, startangle=90, explode=explode)
        ax2.axis("equal")
    st.pyplot(fig2)

# -------------------------------
# Dead Alert Logic
# -------------------------------
def check_dead_alerts(soc, capacity, action, temperature):
    alerts = []
    if soc <= 0.05 * capacity:
        alerts.append("Critical Low SOC")
    if soc >= 0.95 * capacity:
        alerts.append("Critical High SOC")
    if temperature > 40:
        alerts.append("High Temperature")
    if temperature < 5:
        alerts.append("Low Temperature")
    if action == "Idle" and soc == 0:
        alerts.append("Dead Alert: Battery cannot supply demand")
    return alerts

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔋 Battery Dead Alerts in Wind Energy System")

capacity = st.sidebar.slider("Battery Capacity (MWh)", 1, 10, 5, key="capacity_slider")
soc_init = st.sidebar.slider("Initial SOC (%)", 0, 100, 50, key="soc_slider") / 100
temperature_in = st.sidebar.slider("Temperature (°C)", 0, 50, 25, key="temperature_slider")

# -------------------------------
# Run 24h Simulation
# -------------------------------
if st.button("Run 24h Simulation", key="simulate_button"):
    hours = np.arange(24)
    soc_values = []
    actions = []
    alerts_all = []
    soc = soc_init * capacity

    for h in hours:
        # simple wind/demand simulation
        wind = np.random.randint(50, 150)
        load = np.random.randint(80, 130)

        # simplified control: charge if wind > load, discharge otherwise
        if wind > load and soc < capacity:
            action = "Charge"
            soc = min(capacity, soc + (wind - load) * 0.01)
        elif load > wind and soc > 0:
            action = "Discharge"
            soc = max(0, soc - (load - wind) * 0.01)
        else:
            action = "Idle"

        soc_values.append(soc)
        actions.append(action)
        alerts = check_dead_alerts(soc, capacity, action, temperature_in)
        alerts_all.append(", ".join(alerts) if alerts else "Safe")

    df = pd.DataFrame({"Hour": hours, "SOC (MWh)": soc_values, "Action": actions, "Alerts": alerts_all})

    # -------------------------------
    # Graph Visualization with Dead Alerts
    # -------------------------------
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(hours, soc_values, marker='o', color="blue", label="SOC Trajectory")

    # Highlight points with alerts
    for h, soc, alert in zip(hours, soc_values, alerts_all):
        if alert != "Safe":
            ax.plot(h, soc, marker='o', color="red", markersize=10)
            ax.text(h, soc+0.1, "⚠", color="red", fontsize=12, ha="center")

    # Safe zone (green)
    ax.axhspan(0.2*capacity, 0.8*capacity, color="green", alpha=0.2, label="Safe Zone")
    # Warning zones (yellow)
    ax.axhspan(0.05*capacity, 0.2*capacity, color="yellow", alpha=0.2, label="Low Warning")
    ax.axhspan(0.8*capacity, 0.95*capacity, color="yellow", alpha=0.2, label="High Warning")
    # Critical zones (red)
    ax.axhspan(0, 0.05*capacity, color="red", alpha=0.2, label="Critical Low")
    ax.axhspan(0.95*capacity, capacity, color="red", alpha=0.2, label="Critical High")

    ax.set_xlabel("Hour")
    ax.set_ylabel("SOC (MWh)")
    ax.set_title("Battery SOC Trajectory with Dead Alert Zones")
    ax.legend(loc="upper right")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Line-wise Simulation Results")
    st.dataframe(df)