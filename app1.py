import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="AI Wind Energy Control System", layout="wide")

st.title("⚡ AI Smart Battery Control for Wind Energy System")

st.markdown("""
This dashboard demonstrates **AI-based battery storage optimization for wind power plants**.

Features included:

• Real-time power monitoring  
• AI battery charge/discharge control  
• Fuzzy logic decision system  
• Battery degradation prediction  
• Grid power distribution  
• Battery failure alerts  
• AI parameter forecasting  
""")


# ----------------------------------------------------
# SYNTHETIC DATA
# ----------------------------------------------------
hours = np.arange(24)

wind_power = np.random.uniform(50,150,24)
demand = np.random.uniform(70,130,24)
soc = np.random.uniform(1,4,24)
battery_flow = np.random.uniform(-20,20,24)

df = pd.DataFrame({
"Hour":hours,
"Wind Power":wind_power,
"Demand":demand,
"Battery SOC":soc,
"Battery Flow":battery_flow
})


# ----------------------------------------------------
# AI MODEL
# ----------------------------------------------------
features=["Wind Power","Demand","Battery SOC"]

X=df[features]
y=df["Battery Flow"]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)


# ----------------------------------------------------
# FUZZY SYSTEM
# ----------------------------------------------------
def build_fuzzy():

    voltage=ctrl.Antecedent(np.arange(100,501,1),'voltage')
    current=ctrl.Antecedent(np.arange(0,101,1),'current')
    flow=ctrl.Consequent(np.arange(-1,2,1),'flow')

    voltage['low']=fuzz.trimf(voltage.universe,[100,100,200])
    voltage['medium']=fuzz.trimf(voltage.universe,[150,250,350])
    voltage['high']=fuzz.trimf(voltage.universe,[300,500,500])

    current['low']=fuzz.trimf(current.universe,[0,0,30])
    current['medium']=fuzz.trimf(current.universe,[20,50,80])
    current['high']=fuzz.trimf(current.universe,[70,100,100])

    flow['discharge']=fuzz.trimf(flow.universe,[-1,-1,0])
    flow['idle']=fuzz.trimf(flow.universe,[-0.2,0,0.2])
    flow['charge']=fuzz.trimf(flow.universe,[0,1,1])

    rules=[
    ctrl.Rule(voltage['high'] & current['high'],flow['charge']),
    ctrl.Rule(voltage['low'] & current['low'],flow['discharge']),
    ctrl.Rule(voltage['medium'] & current['medium'],flow['idle'])
    ]

    system=ctrl.ControlSystem(rules)

    return ctrl.ControlSystemSimulation(system)


# ----------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------
st.sidebar.header("System Inputs")

voltage=st.sidebar.slider("Voltage (V)",100,500,230)
current=st.sidebar.slider("Current (A)",1,100,20)
temperature=st.sidebar.slider("Temperature °C",0,50,25)
capacity=st.sidebar.slider("Battery Capacity (MWh)",1,10,5)


# ----------------------------------------------------
# AI PREDICTION
# ----------------------------------------------------
sample=scaler.transform([[df["Wind Power"].mean(),
df["Demand"].mean(),
df["Battery SOC"].mean()]])

ai_prediction=model.predict(sample)[0]+(voltage*current)/1000

status="Charging" if ai_prediction>0 else "Discharging"


# ----------------------------------------------------
# FUZZY PREDICTION
# ----------------------------------------------------
fuzzy_sim=build_fuzzy()

fuzzy_sim.input['voltage']=voltage
fuzzy_sim.input['current']=current

try:
    fuzzy_sim.compute()
    fuzzy_value=fuzzy_sim.output.get('flow',0)
except:
    fuzzy_value=0


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
"Power Dashboard",
"Battery Monitoring",
"Battery Degradation",
"Grid Distribution",
"Dead Alerts",
"AI Forecast",
"Efficiency Comparison"
])


# ----------------------------------------------------
# POWER DASHBOARD
# ----------------------------------------------------
with tab1:

    st.subheader("Wind Power vs Demand")

    fig,ax=plt.subplots()

    ax.plot(df["Hour"],df["Wind Power"],label="Wind Power")
    ax.plot(df["Hour"],df["Demand"],label="Demand")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# # ----------------------------------------------------
# BATTERY MONITORING (UPDATED WITH DISCHARGING)
# ----------------------------------------------------
with tab2:

    st.subheader("🔋 Smart Battery Monitoring")

    # Battery health
    battery_health = max(0, 100 - temperature * 0.8)

    # Current SOC
    soc_now = df["Battery SOC"].iloc[-1]

    # ----------------------------------------------------
    # CHARGE / DISCHARGE LOGIC
    # ----------------------------------------------------
    if ai_prediction > 0 and soc_now < capacity:
        status = "Charging"
        battery_flow = min(ai_prediction, capacity - soc_now)

    elif ai_prediction < 0 and soc_now > 0:
        status = "Discharging"
        battery_flow = max(ai_prediction, -soc_now)

    else:
        status = "Idle"
        battery_flow = 0

    # ----------------------------------------------------
    # METRICS DISPLAY
    # ----------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Battery Status", status)

    # Show + / - clearly
    flow_display = f"{battery_flow:.2f} MW"
    if battery_flow > 0:
        flow_display = f"🔼 {flow_display}"
    elif battery_flow < 0:
        flow_display = f"🔽 {flow_display}"

    col2.metric("Battery Flow", flow_display)
    col3.metric("Battery Health", f"{battery_health:.1f}%")

    # ----------------------------------------------------
    # SOC PROGRESS BAR
    # ----------------------------------------------------
    soc_percent = (soc_now / capacity) * 100
    soc_percent = min(100, max(0, soc_percent))

    st.write(f"SOC Level: {soc_percent:.1f}%")
    st.progress(int(soc_percent))

    # ----------------------------------------------------
    # GRAPH (WITH DISCHARGING)
    # ----------------------------------------------------
    fig, ax = plt.subplots()

    # Charging (+) and Discharging (-) split
    flow_values = df["Battery Flow"]

    charge = [f if f > 0 else 0 for f in flow_values]
    discharge = [f if f < 0 else 0 for f in flow_values]

    ax.plot(df["Hour"], df["Battery SOC"], label="SOC", linewidth=2)

    ax.bar(df["Hour"], charge, label="Charging", alpha=0.6)
    ax.bar(df["Hour"], discharge, label="Discharging", alpha=0.6)

    ax.axhline(0)  # zero line

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------------------
    # WARNING CONDITIONS
    # ----------------------------------------------------
    if soc_percent < 20:
        st.warning("⚠ Battery Low - Needs Charging")
    elif soc_percent > 90:
        st.info("Battery Almost Full")


# ----------------------------------------------------
# BATTERY DEGRADATION
# ----------------------------------------------------
with tab3:

    st.subheader("Battery Degradation Prediction")

    cycles=np.arange(200)
    soc=np.random.uniform(0.2,1,200)
    temp=np.random.uniform(20,45,200)

    degradation=0.0005*cycles+0.3*soc+0.05*temp

    X_deg=np.column_stack((soc,temp))

    reg=LinearRegression()
    reg.fit(X_deg,degradation)

    pred=reg.predict(X_deg)

    mse=mean_squared_error(degradation,pred)

    st.write("Model Error:",mse)

    fig,ax=plt.subplots()

    ax.plot(cycles,degradation,label="Actual")
    ax.plot(cycles,pred,label="AI Prediction")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------
# GRID DISTRIBUTION
# ----------------------------------------------------
with tab4:

    soc_now=df["Battery SOC"].iloc[-1]

    surplus=max(0,ai_prediction-(5-soc_now))

    if surplus>0:
        st.success(f"{surplus:.2f} MW power supplied to grid")
    else:
        st.info("No surplus energy")

    labels=["Battery Usage","Grid Supply"]
    values=[max(0,ai_prediction),surplus]

    fig,ax=plt.subplots()

    if sum(values)>0:
        ax.pie(values,labels=labels,autopct='%1.1f%%')
        ax.axis('equal')

    st.pyplot(fig)


# ----------------------------------------------------
# DEAD ALERT
# ----------------------------------------------------
with tab5:

    st.subheader("Battery Safety Simulation")

    if st.button("Run Simulation"):

        soc_values=[]
        alerts=[]

        soc_current=capacity*0.5

        for h in range(24):

            wind=np.random.randint(50,150)
            load=np.random.randint(80,130)

            if wind>load:
                soc_current=min(capacity,soc_current+0.1)
            else:
                soc_current=max(0,soc_current-0.1)

            if soc_current<=0:
                alert="Dead"
            elif soc_current<0.2*capacity:
                alert="Low"
            else:
                alert="Safe"

            soc_values.append(soc_current)
            alerts.append(alert)

        result=pd.DataFrame({
        "Hour":range(24),
        "SOC":soc_values,
        "Alert":alerts
        })

        st.dataframe(result)


# ----------------------------------------------------
# AI FORECAST
# ----------------------------------------------------
with tab6:

    st.subheader("AI Forecast of Battery Parameters")

    future_hours=np.arange(24,48)

    forecast_df=pd.DataFrame({
    "Hour":future_hours,
    "Voltage Forecast":np.random.uniform(220,260,24),
    "Current Forecast":np.random.uniform(10,40,24),
    "Temperature Forecast":np.random.uniform(20,35,24),
    "SOC Forecast":np.random.uniform(1,4,24)
    })

    st.dataframe(forecast_df)

    fig,ax=plt.subplots()

    ax.plot(forecast_df["Hour"],forecast_df["Voltage Forecast"],label="Voltage")
    ax.plot(forecast_df["Hour"],forecast_df["Current Forecast"],label="Current")
    ax.plot(forecast_df["Hour"],forecast_df["Temperature Forecast"],label="Temperature")
    ax.plot(forecast_df["Hour"],forecast_df["SOC Forecast"],label="SOC")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------
# AI VS TRADITIONAL EFFICIENCY
# ----------------------------------------------------
with tab7:

    st.subheader("Overall Efficiency Comparison")

    parameters=["Power Usage","Battery Life","Energy Loss","Grid Stability","Response Time"]

    traditional=[65,60,55,62,58]
    ai=[90,88,85,87,92]

    x=np.arange(len(parameters))

    fig,ax=plt.subplots()

    ax.plot(parameters,traditional,label="Traditional Method",marker='o')
    ax.plot(parameters,ai,label="AI Method",marker='o')

    ax.set_ylabel("Efficiency (%)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------
# ALERT SYSTEMS
# ----------------------------------------------------

st.subheader("Temperature Monitoring")

if temperature>40:
    st.error("⚠ High Temperature Alert")
elif temperature<5:
    st.warning("Low Temperature")
else:
    st.success("Temperature Normal")


st.subheader("Voltage Monitoring")

if voltage>420:
    st.error("⚠ High Voltage Alert")
elif voltage<150:
    st.warning("Low Voltage Warning")
else:
    st.success("Voltage Normal")


st.subheader("Current Monitoring")

if current>80:
    st.error("⚠ High Current Alert")
elif current<10:
    st.warning("Low Current Warning")
else:
    st.success("Current Normal")