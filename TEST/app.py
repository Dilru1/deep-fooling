import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re

# --- TensorBoard Integration ---
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# --- Configuration ---
DIR_EVAL_MLP = Path("MLP")          
DIR_EVAL_CNN = Path("1DCNN")    
DIR_EVAL_PID = Path("Baseline") 

# Training Log Paths (Adjust these to where your tfevents files live)
DIR_TRAIN_MLP = Path("../model/Par_100000_MLP")
DIR_TRAIN_CNN = Path("../model/Par_100000_1DCNN")

PAGE_TITLE = "Sailboat Performance Dashboard"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- Helper Functions ---

@st.cache_data
def load_tb_logs(base_dir, model_label):
    if not TB_AVAILABLE or not base_dir.exists(): return pd.DataFrame()
    log_files = list(base_dir.glob("**/events.out.tfevents*"))
    all_records = []
    for log_file in log_files:
        try:
            event_acc = EventAccumulator(str(log_file))
            event_acc.Reload()
            for tag in event_acc.Tags()['scalars']:
                if tag in ['rollout/ep_rew_mean', 'train/loss']:
                    for e in event_acc.Scalars(tag):
                        all_records.append({
                            "Model": model_label, "Metric": tag,
                            "Step": e.step, "Value": e.value, "File": log_file.name
                        })
        except: continue
    return pd.DataFrame(all_records)

@st.cache_data
def load_eval_data():
    configs = [(DIR_EVAL_MLP, "MLP"), (DIR_EVAL_CNN, "CNN"), (DIR_EVAL_PID, "PID")]
    all_dfs = []
    # Pattern matching your specific filename structure: heading_X_wind_Y
    pattern = re.compile(r"heading_([\d\.]+)_wind_([\d\.]+)")
    for directory, label in configs:
        if not directory.exists(): continue
        for f in directory.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                match = pattern.search(f.name)
                if match:
                    df['Model'], df['Filename'] = label, f.name
                    df['Target Heading'] = float(match.group(1))
                    df['Wind Speed'] = float(match.group(2))
                    # VMG Calculation
                    df['VMG'] = df['sog_knots'] * np.cos(np.radians(df['cmg_deg'] - df['Target Heading']))
                    all_dfs.append(df)
            except: continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- App Logic ---

df_eval = load_eval_data()

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Global Eval (Histograms)", "Training Curves", "Single Inspector"])

# --- VIEW 1: HISTOGRAMS & TRAJECTORIES ---
if app_mode == "Global Eval (Histograms)":
    st.subheader("Performance Distributions")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_sog = px.histogram(df_eval, x="sog_knots", color="Model", marginal="box", 
                               barmode="overlay", title="SOG (Speed Over Ground) Distribution")
        st.plotly_chart(fig_sog, use_container_width=True)
    with col2:
        fig_vmg = px.histogram(df_eval, x="VMG", color="Model", marginal="box", 
                               barmode="overlay", title="VMG (Velocity Made Good) Distribution")
        st.plotly_chart(fig_vmg, use_container_width=True)

    st.subheader("Global Trajectory Overlay")
    fig_map = px.line(df_eval, x="pos_x", y="pos_y", color="Model", line_group="Filename",
                      hover_data=["Target Heading", "Wind Speed"])
    fig_map.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), height=700)
    st.plotly_chart(fig_map, use_container_width=True)

# --- VIEW 2: TRAINING CURVES ---
elif app_mode == "Training Curves":
    st.subheader("TensorBoard Training Progress")
    df_tb = pd.concat([load_tb_logs(DIR_TRAIN_MLP, "MLP"), load_tb_logs(DIR_TRAIN_CNN, "CNN")])
    
    if df_tb.empty:
        st.warning("No TensorBoard logs found. Check DIR_TRAIN paths.")
    else:
        metric = st.selectbox("Select Metric", df_tb['Metric'].unique())
        fig_train = px.line(df_tb[df_tb['Metric']==metric], x="Step", y="Value", color="Model", 
                            line_group="File", title=f"{metric} over Training Steps")
        fig_train.update_traces(opacity=0.7)
        st.plotly_chart(fig_train, use_container_width=True)

# --- VIEW 3: SINGLE INSPECTOR ---
elif app_mode == "Single Inspector":
    st.subheader("Episode Deep Dive")
    model_choice = st.selectbox("Model", df_eval['Model'].unique())
    file_choice = st.selectbox("Episode", df_eval[df_eval['Model']==model_choice]['Filename'].unique())
    
    df_sub = df_eval[df_eval['Filename']==file_choice].sort_values("step")
    
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]],
                        subplot_titles=("Trajectory", "SOG vs Time", "Action Offset"))
    
    fig.add_trace(go.Scatter(x=df_sub['pos_x'], y=df_sub['pos_y'], name="Path"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sub['step'], y=df_sub['sog_knots'], name="SOG"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_sub['step'], y=df_sub['action_offset'], name="Action"), row=2, col=2)
    
    fig.update_layout(height=700)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)