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
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
"Power Dashboard",
"Battery Monitoring",
"Battery Degradation",
"Grid Distribution",
"Dead Alerts",
"AI Forecast"
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


# ----------------------------------------------------
# PROFESSIONAL BATTERY MONITORING
# ----------------------------------------------------
with tab2:

    st.subheader("🔋 Smart Battery Monitoring System")

    battery_health=max(0,100-temperature*0.8)

    col1,col2,col3=st.columns(3)

    col1.metric("Battery Status",status)
    col2.metric("AI Battery Flow",f"{ai_prediction:.2f} MW")
    col3.metric("Battery Health",f"{battery_health:.1f}%")

    st.divider()

    soc_now=df["Battery SOC"].iloc[-1]
    soc_percent=(soc_now/capacity)*100
    soc_percent=min(100,soc_percent)

    st.write("### Battery State of Charge")

    st.progress(int(soc_percent))
    st.write(f"SOC Level: **{soc_percent:.1f}%**")

    if soc_percent<20:
        st.error("⚠ Battery Critical Level")
    elif soc_percent<40:
        st.warning("Battery Low")
    else:
        st.success("Battery Operating Normally")

    st.divider()

    st.write("### Battery Performance Over Time")

    fig,ax=plt.subplots(figsize=(10,4))

    ax.plot(df["Hour"],df["Battery SOC"],label="SOC",linewidth=3)
    ax.plot(df["Hour"],df["Battery Flow"],label="Battery Flow",linestyle="--")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Energy Level")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


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

    degradation_df=pd.DataFrame({
    "Cycle":cycles,
    "SOC":soc,
    "Temperature":temp,
    "Actual Degradation":degradation,
    "Predicted Degradation":pred
    })

    st.dataframe(degradation_df)

    fig,ax=plt.subplots()

    ax.plot(cycles,degradation,label="Actual")
    ax.plot(cycles,pred,label="AI Prediction")

    ax.set_xlabel("Charge Cycles")
    ax.set_ylabel("Degradation")
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
        colors=[]

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
                colors.append("red")
            elif soc_current<0.2*capacity:
                alert="Low"
                colors.append("orange")
            else:
                alert="Safe"
                colors.append("green")

            soc_values.append(soc_current)
            alerts.append(alert)

        result=pd.DataFrame({
        "Hour":range(24),
        "SOC":soc_values,
        "Alert":alerts
        })

        st.dataframe(result)

        fig,ax=plt.subplots()

        ax.plot(result["Hour"],result["SOC"],color="black")

        for i in range(len(result)):
            ax.scatter(result["Hour"][i],result["SOC"][i],color=colors[i],s=100)

        ax.set_xlabel("Hour")
        ax.set_ylabel("Battery SOC")
        ax.grid(True)

        st.pyplot(fig)


# ----------------------------------------------------
# AI FORECAST
# ----------------------------------------------------
with tab6:

    st.subheader("AI Forecast of Battery Parameters")

    history_hours=np.arange(24)

    voltage_hist=np.random.uniform(200,260,24)
    current_hist=np.random.uniform(10,40,24)
    temp_hist=np.random.uniform(20,35,24)
    soc_hist=np.random.uniform(1,4,24)

    history_df=pd.DataFrame({
    "Hour":history_hours,
    "Voltage":voltage_hist,
    "Current":current_hist,
    "Temperature":temp_hist,
    "SOC":soc_hist
    })

    X_train=history_df[["Hour"]]

    model_v=LinearRegression()
    model_c=LinearRegression()
    model_t=LinearRegression()
    model_s=LinearRegression()

    model_v.fit(X_train,voltage_hist)
    model_c.fit(X_train,current_hist)
    model_t.fit(X_train,temp_hist)
    model_s.fit(X_train,soc_hist)

    future_hours=np.arange(24,48)

    forecast_df=pd.DataFrame({"Hour":future_hours})

    forecast_df["Voltage Forecast"]=model_v.predict(forecast_df[["Hour"]])
    forecast_df["Current Forecast"]=model_c.predict(forecast_df[["Hour"]])
    forecast_df["Temperature Forecast"]=model_t.predict(forecast_df[["Hour"]])
    forecast_df["SOC Forecast"]=model_s.predict(forecast_df[["Hour"]])

    st.dataframe(forecast_df)

    fig,ax=plt.subplots()

    ax.plot(forecast_df["Hour"],forecast_df["Voltage Forecast"],label="Voltage")
    ax.plot(forecast_df["Hour"],forecast_df["Current Forecast"],label="Current")
    ax.plot(forecast_df["Hour"],forecast_df["Temperature Forecast"],label="Temperature")
    ax.plot(forecast_df["Hour"],forecast_df["SOC Forecast"],label="SOC")

    ax.set_xlabel("Future Hours")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ----------------------------------------------------
# TEMPERATURE ALERT
# ----------------------------------------------------
st.subheader("Temperature Monitoring")

if temperature>40:
    st.error("High Temperature Alert")
elif temperature<5:
    st.warning("Low Temperature")
else:
    st.success("Temperature Normal")