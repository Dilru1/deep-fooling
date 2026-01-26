import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import os

# --- Import TensorBoard Parser ---
# Fallback if tensorboard is not installed
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# --- Configuration ---
# 1. EVALUATION DATA (The CSVs you generated)
DIR_EVAL_MLP = Path("Output_CSV")          
DIR_EVAL_CNN = Path("Output_CSV_1DCNN")    
DIR_EVAL_PID = Path("Output_Baseline")     

# 2. TRAINING LOGS (The original model folders containing 'events.out.tfevents...')
# Update these paths to point to your actual model folders
DIR_TRAIN_MLP = Path("../model/Par_100000_MLP")
DIR_TRAIN_CNN = Path("../model/Par_100000_1DCNN")

PAGE_TITLE = "Boat"
LAYOUT = "wide"
WARMUP_STEPS = 50

st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

# --- Helper Functions ---

@st.cache_data
def load_tensorboard_data(base_dir, model_label):
    """
    Crawls a directory for 'events.out.tfevents' files, parses them,
    and returns a DataFrame of training metrics.
    """
    if not TB_AVAILABLE:
        return pd.DataFrame()
    
    if not base_dir.exists():
        return pd.DataFrame()

    # Find all tfevents files recursively
    # Structure usually: Model_Dir / seed_0 / events.out.tfevents...
    log_files = list(base_dir.glob("**/events.out.tfevents*"))
    
    all_records = []
    
    # Progress bar for loading logs (can be slow)
    prog_text = f"Parsing {model_label} TensorBoard logs..."
    my_bar = st.progress(0, text=prog_text)
    
    for i, log_file in enumerate(log_files):
        try:
            # Extract Seed from path (assuming folder name contains 'seed')
            seed = "unknown"
            for part in log_file.parts:
                if "seed" in part:
                    seed = part
                    break
            
            # Load the EventAccumulator
            event_acc = EventAccumulator(str(log_file))
            event_acc.Reload()
            
            # Extract specific tags we care about
            # Common SB3 tags: 'rollout/ep_rew_mean', 'train/loss', 'train/value_loss'
            tags = event_acc.Tags()['scalars']
            
            for tag in tags:
                # We usually care most about reward
                if tag in ['rollout/ep_rew_mean', 'train/loss', 'train/entropy_loss']:
                    events = event_acc.Scalars(tag)
                    for e in events:
                        all_records.append({
                            "Model": model_label,
                            "Seed": seed,
                            "Metric": tag,
                            "Timestep": e.step,
                            "Value": e.value
                        })
        except Exception as e:
            continue
            
        my_bar.progress((i + 1) / len(log_files), text=prog_text)
        
    my_bar.empty()
    
    return pd.DataFrame(all_records)

@st.cache_data
def load_eval_dataset(directory, model_label):
    # ... (Same CSV loading logic as before) ...
    if not directory.exists(): return pd.DataFrame()
    files = sorted(list(directory.glob("*.csv")))
    if not files: return pd.DataFrame()
    data_list = []
    pattern = re.compile(r"heading_([\d\.]+)_wind_([\d\.]+)")
    seed_pattern = re.compile(r"seed_(\w+)") 
    
    for f in files:
        try:
            df = pd.read_csv(f)
            match = pattern.search(f.name)
            if match:
                heading = float(match.group(1))
                wind = float(match.group(2))
                seed_match = seed_pattern.search(f.name)
                seed = seed_match.group(1) if seed_match else "base"
                df['Model'] = model_label
                df['Seed'] = seed
                df['Target Heading'] = heading
                df['Wind Speed'] = wind
                df['ID'] = f"{model_label}_{seed}_H{int(heading)}_W{int(wind)}"
                df['Filename'] = f.name
                if len(df) > WARMUP_STEPS:
                    avg_cmg = df.iloc[WARMUP_STEPS:]['cmg_env'].mean()
                else:
                    avg_cmg = df['cmg_env'].mean()
                df['Avg_CMG_Run'] = avg_cmg
                data_list.append(df)
        except: continue
    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def parse_filename(filename):
    match = re.search(r"heading_([\d\.]+)_wind_([\d\.]+)", filename)
    if match: return f"Heading: {match.group(1)}° | Wind: {match.group(2)} kn"
    return filename

# --- Main App ---

st.title(f"{PAGE_TITLE}")

# Sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to:", [
    "Global Dashboard (Evaluation)", 
    "Training Curves (TensorBoard)",  # <--- NEW OPTION
    "Single Inspector (Deep Dive)"
])

# --- VIEW 1: GLOBAL DASHBOARD (Evaluation CSVs) ---
if app_mode == "Global Dashboard (Evaluation)":
    # (Same code as previous turn for Bar Charts and Maps)
    with st.spinner("Loading evaluation CSVs..."):
        df_mlp = load_eval_dataset(DIR_EVAL_MLP, "MLP")
        df_cnn = load_eval_dataset(DIR_EVAL_CNN, "CNN")
        df_pid = load_eval_dataset(DIR_EVAL_PID, "PID")
        
    if df_mlp.empty and df_cnn.empty and df_pid.empty:
        st.warning("No evaluation data found.")
        st.stop()
        
    df_master = pd.concat([df_mlp, df_cnn, df_pid], ignore_index=True)
    
    st.markdown("### Avergae CMG")
    tab1, tab2 = st.tabs(["Stats", "Trajectories"])
    
    with tab1:
        st.subheader("Steady State CMG")
        df_summary = df_master.groupby(['Model', 'Target Heading', 'Wind Speed'])['Avg_CMG_Run'].agg(['mean', 'std']).reset_index()
        color_map = {"MLP": "#636EFA", "CNN": "#00CC96", "PID": "#EF553B"}
        fig_bar = px.bar(
            df_summary, x="Target Heading", y="mean", color="Model",
            error_y="std", facet_col="Wind Speed", barmode="group",
            color_discrete_map=color_map, category_orders={"Model": ["CNN", "MLP", "PID"]}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Improvement Table
        pivot_df = df_summary.pivot(index=['Target Heading', 'Wind Speed'], columns='Model', values='mean')
        if 'PID' in pivot_df.columns:
            if 'MLP' in pivot_df.columns: pivot_df['MLP Gain %'] = ((pivot_df['MLP'] - pivot_df['PID']) / pivot_df['PID']) * 100
            if 'CNN' in pivot_df.columns: pivot_df['CNN Gain %'] = ((pivot_df['CNN'] - pivot_df['PID']) / pivot_df['PID']) * 100
            cols = [c for c in ['PID', 'MLP', 'CNN', 'MLP Gain %', 'CNN Gain %'] if c in pivot_df.columns]
            st.dataframe(pivot_df[cols].style.format("{:.2f}").background_gradient(subset=[c for c in cols if 'Gain' in c], cmap="RdYlGn"))

    with tab2:
        st.subheader("Trajectory Overlay")
        # (Filtering and map logic from previous turn)
        col1, col2 = st.columns(2)
        avail_winds = sorted(df_master['Wind Speed'].unique())
        sel_winds = col1.multiselect("Filter Wind:", avail_winds, default=avail_winds[:1])
        df_map = df_master[df_master['Wind Speed'].isin(sel_winds)]
        
        fig_map = px.line(
            df_map, x="pos_x", y="pos_y", color="Model", line_dash="Target Heading",
            line_group="ID", color_discrete_map={"MLP": "#636EFA", "CNN": "#00CC96", "PID": "#EF553B"}
        )
        fig_map.update_layout(height=600, yaxis=dict(scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig_map, use_container_width=True)

# --- VIEW 2: TRAINING CURVES (NEW) ---
elif app_mode == "Training Curves (TensorBoard)":
    st.markdown("### Training Progress (TensorBoard Logs)")
    
    if not TB_AVAILABLE:
        st.error("`tensorboard` library not found. Please install via `pip install tensorboard`.")
        st.stop()

    with st.spinner("Parsing TensorBoard logs... this may take a moment."):
        df_tb_mlp = load_tensorboard_data(DIR_TRAIN_MLP, "MLP")
        df_tb_cnn = load_tensorboard_data(DIR_TRAIN_CNN, "CNN")
    
    df_train = pd.concat([df_tb_mlp, df_tb_cnn], ignore_index=True)
    
    if df_train.empty:
        st.warning(f"No training logs found in {DIR_TRAIN_MLP} or {DIR_TRAIN_CNN}.")
        st.stop()

    # User Controls
    available_metrics = df_train['Metric'].unique()
    
    # Set default to reward if available
    default_ix = list(available_metrics).index('rollout/ep_rew_mean') if 'rollout/ep_rew_mean' in available_metrics else 0
    selected_metric = st.selectbox("Select Metric:", available_metrics, index=default_ix)
    
    # Filter Data
    df_plot = df_train[df_train['Metric'] == selected_metric]
    
    # Plotting
    # We use a line plot. If multiple seeds exist, we can aggregate them or show all lines.
    st.subheader(f"{selected_metric} over Time")
    
    tab_raw, tab_agg = st.tabs(["Individual Seeds", "Aggregated (Mean + Std)"])
    
    with tab_raw:
        # Show every seed as a separate line
        fig_raw = px.line(
            df_plot, 
            x="Timestep", 
            y="Value", 
            color="Model", 
            line_group="Seed", # Makes separate lines for seeds of same model
            title=f"{selected_metric} (Raw)",
            color_discrete_map={"MLP": "#636EFA", "CNN": "#00CC96"}
        )
        
        # FIX: Apply opacity here instead of inside px.line
        fig_raw.update_traces(opacity=0.6)
        
        st.plotly_chart(fig_raw, use_container_width=True)
                        
    with tab_agg:
        # Aggregate by Model and binning steps (to make smoothed curves)
        # Simple approach: Plotly trendlines or simple pandas rolling mean
        # Here we rely on visual aggregation
        fig_agg = px.scatter(
            df_plot,
            x="Timestep", 
            y="Value", 
            color="Model",
            trendline="lowess", # Locally Weighted Scatterplot Smoothing
            trendline_options=dict(frac=0.1),
            title=f"{selected_metric} (Smoothed Trend)",
            color_discrete_map={"MLP": "#636EFA", "CNN": "#00CC96"},
        )
        # Hide the scatter points to just show trendlines? 
        # Usually better to show both or just lines. 
        # Let's stick to standard line plot but tell user to look at general trend
        st.plotly_chart(fig_agg, use_container_width=True)
        st.caption("Note: 'Lowess' trendline fits a smooth curve through all seed data points.")

# --- VIEW 3: SINGLE INSPECTOR ---
elif app_mode == "Single Inspector (Deep Dive)":
    # (Same code as previous turn)
    st.header("Deep Dive Inspector")
    dataset_choice = st.radio("Select Dataset:", ["MLP Results", "1D-CNN Results", "PID Baseline"], horizontal=True)
    
    if dataset_choice == "MLP Results": active_dir = DIR_EVAL_MLP
    elif dataset_choice == "1D-CNN Results": active_dir = DIR_EVAL_CNN
    else: active_dir = DIR_EVAL_PID
    
    if not active_dir.exists(): st.stop()
    files = sorted([f.name for f in list(active_dir.glob("*.csv"))])
    if not files: st.stop()
        
    selected_file = st.selectbox("Select File:", files, format_func=lambda x: f"{x} ({parse_filename(x)})")
    df = pd.read_csv(active_dir / selected_file)
    
    # ... (Metrics and Plot code from previous turn) ...
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df['step'], y=df['ground_speed'], name="Speed", line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['step'], y=df['cmg_env'], name="CMG", line=dict(color='orange', dash='dash')), row=1, col=1)
    if 'course_relative' in df.columns: fig.add_trace(go.Scatter(x=df['step'], y=df['course_relative'], name="Error", line=dict(color='red')), row=2, col=1)
    if 'action' in df.columns: fig.add_trace(go.Scatter(x=df['step'], y=df['action'], name="Action", line=dict(color='blue', opacity=0.5)), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)