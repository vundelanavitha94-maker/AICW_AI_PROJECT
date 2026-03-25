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


# ----------------------------------------------------
# SYNTHETIC DATA
# ----------------------------------------------------
hours = np.arange(24)

wind_power = np.random.uniform(50,150,24)
demand = np.random.uniform(70,130,24)
soc = np.random.uniform(1,5,24)

df = pd.DataFrame({
    "Hour":hours,
    "Wind Power":wind_power,
    "Demand":demand,
    "Battery SOC":soc
})


# ----------------------------------------------------
# AI MODEL
# ----------------------------------------------------
features=["Wind Power","Demand","Battery SOC"]

X=df[features]
y = wind_power - demand   # better logic

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
    soc=ctrl.Antecedent(np.arange(0,101,1),'soc')

    flow=ctrl.Consequent(np.arange(-1,2,1),'flow')

    voltage['low']=fuzz.trimf(voltage.universe,[100,100,200])
    voltage['high']=fuzz.trimf(voltage.universe,[300,500,500])

    current['low']=fuzz.trimf(current.universe,[0,0,30])
    current['high']=fuzz.trimf(current.universe,[70,100,100])

    soc['low']=fuzz.trimf(soc.universe,[0,0,40])
    soc['high']=fuzz.trimf(soc.universe,[60,100,100])

    flow['discharge']=fuzz.trimf(flow.universe,[-1,-1,0])
    flow['idle']=fuzz.trimf(flow.universe,[-0.2,0,0.2])
    flow['charge']=fuzz.trimf(flow.universe,[0,1,1])

    rules=[
        ctrl.Rule(voltage['high'] & soc['low'], flow['charge']),
        ctrl.Rule(voltage['low'] & soc['high'], flow['discharge']),
        ctrl.Rule(current['low'], flow['idle'])
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

ai_output=model.predict(sample)[0]

soc_now=df["Battery SOC"].iloc[-1]

# ----------------------------------------------------
# CHARGE / DISCHARGE LOGIC
# ----------------------------------------------------
if ai_output > 0 and soc_now < capacity:
    battery_action = "Charging"
    battery_flow = min(ai_output, capacity - soc_now)

elif ai_output < 0 and soc_now > 0:
    battery_action = "Discharging"
    battery_flow = max(ai_output, -soc_now)

else:
    battery_action = "Idle"
    battery_flow = 0


# ----------------------------------------------------
# FUZZY OUTPUT
# ----------------------------------------------------
fuzzy_sim=build_fuzzy()

fuzzy_sim.input['voltage']=voltage
fuzzy_sim.input['current']=current
fuzzy_sim.input['soc']= (soc_now/capacity)*100

try:
    fuzzy_sim.compute()
    fuzzy_val=fuzzy_sim.output['flow']
except:
    fuzzy_val=0


# ----------------------------------------------------
# EFFICIENCY CALCULATION
# ----------------------------------------------------
efficiency = 90 - (temperature*0.3) - abs(fuzzy_val*10)
efficiency = max(50, min(100, efficiency))


# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1,tab2,tab3,tab4 = st.tabs([
    "Power",
    "Battery",
    "Efficiency",
    "Forecast"
])


# ----------------------------------------------------
# TAB 1 POWER
# ----------------------------------------------------
with tab1:
    fig,ax=plt.subplots()
    ax.plot(df["Hour"],df["Wind Power"],label="Wind Power")
    ax.plot(df["Hour"],df["Demand"],label="Demand")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# ----------------------------------------------------
# TAB 2 BATTERY
# ----------------------------------------------------
with tab2:

    col1,col2,col3=st.columns(3)

    col1.metric("Action",battery_action)
    col2.metric("Battery Flow",f"{battery_flow:.2f} MW")
    col3.metric("Efficiency",f"{efficiency:.1f}%")

    soc_percent=(soc_now/capacity)*100
    st.progress(int(soc_percent))


# ----------------------------------------------------
# TAB 3 EFFICIENCY GRAPH
# ----------------------------------------------------
with tab3:

    temp_range=np.arange(0,50)
    eff_curve=90 - temp_range*0.3

    fig,ax=plt.subplots()
    ax.plot(temp_range,eff_curve)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Efficiency")
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------
# TAB 4 FORECAST
# ----------------------------------------------------
with tab4:

    future=pd.DataFrame({
        "Hour":np.arange(24,48),
        "Wind Forecast":np.random.uniform(60,140,24),
        "Demand Forecast":np.random.uniform(70,130,24)
    })

    st.dataframe(future)


# ----------------------------------------------------
# ALERTS
# ----------------------------------------------------
st.subheader("⚠ Alerts")

if temperature>40:
    st.error("High Temperature!")
if voltage>420:
    st.error("High Voltage!")
if current>80:
    st.error("High Current!")