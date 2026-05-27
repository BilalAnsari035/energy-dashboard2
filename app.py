
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="⚡ Global Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #020817;
    color: #e2e8f0;
}
.main { background-color: #020817; }
.stApp {
    background: linear-gradient(135deg, #020817 0%, #0a1628 50%, #020817 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #051020 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f3c, #051020);
    border: 1px solid #1e40af;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.dashboard-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 10px 0;
    letter-spacing: 2px;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 30px;
}
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #38bdf8;
    padding-left: 12px;
    margin: 20px 0 15px 0;
}
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e40af, transparent);
    margin: 25px 0;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0d1f3c;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: white !important;
}
.pred-highlight {
    background: linear-gradient(135deg, #1a1040, #0d0826);
    border: 1px solid #6d28d9;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(109, 40, 217, 0.2);
}
.pred-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    color: #a78bfa;
    font-weight: 700;
}
.pred-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    df = df[df['year'] >= 1990]
    return df

with st.spinner("⚡ Energy data load ho raha hai..."):
    df = load_data()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-family: Orbitron; font-size: 1.1rem; color: #38bdf8; letter-spacing: 2px;'>⚡ ENERGY INTEL</div>
        <div style='color: #475569; font-size: 0.7rem; letter-spacing: 1px;'>Global Dashboard v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🌍 Countries")
    exclude_kw = ['World','Asia','Africa','Europe','America',
                  'OECD','income','Ember','EI','G20','G7','Union']
    all_c = sorted(df['country'].unique())
    countries = [c for c in all_c if not any(k in c for k in exclude_kw)]

    selected_countries = st.multiselect(
        "Countries chunein",
        options=countries,
        default=["Pakistan", "India", "China", "United States", "Germany"]
    )

    year_range = st.slider(
        "📅 Saal ka Range",
        int(df['year'].min()), int(df['year'].max()), (2000, 2022)
    )

    pred_year = st.slider("🔮 Forecast tak", 2023, 2040, 2030)

    st.markdown("---")
    st.markdown("""
    <div style='background:#0d1f3c; border:1px solid #1e3a5f; border-radius:10px; padding:15px; color:#94a3b8; font-size:0.85rem;'>
    📊 <b>Data:</b> Our World in Data<br><br>
    🔄 <b>Updated:</b> Annually<br><br>
    📈 <b>Coverage:</b> 200+ countries
    </div>
    """, unsafe_allow_html=True)

if not selected_countries:
    st.warning("⚠️ Sidebar se kam az kam ek country select karein!")
    st.stop()

filtered = df[
    (df['country'].isin(selected_countries)) &
    (df['year'].between(year_range[0], year_range[1]))
].copy()

st.markdown('<div class="dashboard-title">⚡ GLOBAL ENERGY INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Our World in Data · Analytics & ML Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

latest_year = df['year'].max()
latest = df[(df['year'] == latest_year) & (df['country'].isin(selected_countries))]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("⚡ Total Energy", f"{latest['primary_energy_consumption'].sum():,.0f} TWh")
with col2:
    st.metric("🌱 Renewables Avg", f"{latest['renewables_share_energy'].mean():.1f}%")
with col3:
    val3 = latest['fossil_share_energy'].mean() if 'fossil_share_energy' in df.columns else 0
    st.metric("🛢️ Fossil Fuel", f"{val3:.1f}%")
with col4:
    st.metric("👤 Energy/Capita", f"{latest['energy_per_capita'].mean():,.0f} kWh")
with col5:
    elec = latest['electricity_generation'].sum() if 'electricity_generation' in df.columns else 0
    st.metric("🔌 Electricity", f"{elec:,.0f} TWh")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Energy Trends",
    "🌍 World Map",
    "🥧 Energy Mix",
    "🌱 Renewables vs Fossil",
    "🔮 ML Predictions"
])

DARK = "plotly_dark"
COLORS = px.colors.qualitative.Bold

with tab1:
    st.markdown('<div class="section-header">Primary Energy Consumption</div>', unsafe_allow_html=True)
    fig = px.line(filtered, x='year', y='primary_energy_consumption',
        color='country', template=DARK, color_discrete_sequence=COLORS,
        labels={'primary_energy_consumption': 'TWh', 'year': 'Year'})
    fig.update_traces(line_width=2.5)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,31,60,0.5)', height=420,
        legend=dict(bgcolor='rgba(0,0,0,0)'), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.bar(filtered, x='year', y='energy_per_capita', color='country',
            template=DARK, color_discrete_sequence=COLORS, barmode='group',
            title='Energy Per Capita (kWh)')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,31,60,0.5)', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        latest_bar = latest.sort_values('primary_energy_consumption', ascending=True)
        fig3 = px.bar(latest_bar, x='primary_energy_consumption', y='country',
            orientation='h', template=DARK, color='primary_energy_consumption',
            color_continuous_scale='Blues', title=f'Total Energy {latest_year}')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,31,60,0.5)', height=350)
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">World Energy Map</div>', unsafe_allow_html=True)
    map_year = st.select_slider("Saal chunein",
        options=sorted(df['year'].unique()), value=2022)
    world_data = df[df['year'] == map_year][
        ['country','primary_energy_consumption','renewables_share_energy','energy_per_capita']].dropna()
    map_metric = st.radio("Metric chunein",
        ["primary_energy_consumption","renewables_share_energy","energy_per_capita"],
        horizontal=True,
        format_func=lambda x: {'primary_energy_consumption':'⚡ Total Energy',
            'renewables_share_energy':'🌱 Renewables %',
            'energy_per_capita':'👤 Per Capita'}[x])
    fig_map = px.choropleth(world_data, locations='country',
        locationmode='country names', color=map_metric,
        color_continuous_scale='Viridis' if 'renewable' in map_metric else 'Blues',
        template=DARK, title=f'{map_metric.replace("_"," ").title()} — {map_year}')
    fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False), height=500)
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Energy Mix Breakdown</div>', unsafe_allow_html=True)
    mix_country = st.selectbox("Country chunein", selected_countries)
    mix_df = df[(df['country'] == mix_country) &
        (df['year'].between(year_range[0], year_range[1]))]
    sources = {'Coal':'coal_share_energy','Oil':'oil_share_energy',
        'Gas':'gas_share_energy','Nuclear':'nuclear_share_energy',
        'Renewables':'renewables_share_energy','Hydro':'hydro_share_energy',
        'Solar':'solar_share_energy','Wind':'wind_share_energy'}
    avail = {k:v for k,v in sources.items() if v in mix_df.columns}
    latest_mix = mix_df[mix_df['year']==mix_df['year'].max()][list(avail.values())].mean()
    latest_mix.index = list(avail.keys())
    latest_mix = latest_mix.dropna()

    col_pie, col_area = st.columns([1,2])
    with col_pie:
        fig_pie = go.Figure(go.Pie(labels=latest_mix.index, values=latest_mix.values,
            hole=0.5, marker=dict(colors=COLORS, line=dict(color='#020817', width=2))))
        fig_pie.update_layout(template=DARK, paper_bgcolor='rgba(0,0,0,0)',
            title=f'{mix_country} — Latest Mix', height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_area:
        area_cols = [v for v in avail.values() if v in mix_df.columns]
        area_data = mix_df[['year']+area_cols].set_index('year')
        area_data.columns = list(avail.keys())[:len(area_cols)]
        fig_area = go.Figure()
        for i, col in enumerate(area_data.columns):
            fig_area.add_trace(go.Scatter(x=area_data.index, y=area_data[col],
                name=col, stackgroup='one', mode='none',
                fillcolor=COLORS[i % len(COLORS)]))
        fig_area.update_layout(template=DARK, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,31,60,0.5)',
            title=f'{mix_country} — Mix Over Time', height=380)
        st.plotly_chart(fig_area, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Renewables vs Fossil Fuel</div>', unsafe_allow_html=True)
    fig_rv = go.Figure()
    for i, country in enumerate(selected_countries):
        c_df = filtered[filtered['country']==country]
        color = COLORS[i % len(COLORS)]
        if 'renewables_share_energy' in c_df.columns:
            fig_rv.add_trace(go.Scatter(x=c_df['year'], y=c_df['renewables_share_energy'],
                name=f'{country} 🌱', line=dict(color=color, width=2),
                mode='lines+markers', marker_size=4))
        if 'fossil_share_energy' in c_df.columns:
            fig_rv.add_trace(go.Scatter(x=c_df['year'], y=c_df['fossil_share_energy'],
                name=f'{country} 🛢️', line=dict(color=color, width=2, dash='dash'),
                mode='lines+markers', marker_size=4))
    fig_rv.update_layout(template=DARK, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,31,60,0.5)',
        title='Renewables (solid) vs Fossil (dashed)',
        yaxis_title='% Share', hovermode='x unified', height=450)
    st.plotly_chart(fig_rv, use_container_width=True)

    if 'greenhouse_gas_emissions' in df.columns:
        fig_co2 = px.area(filtered, x='year', y='greenhouse_gas_emissions',
            color='country', template=DARK, color_discrete_sequence=COLORS,
            title='Greenhouse Gas Emissions (Mt CO₂e)')
        fig_co2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,31,60,0.5)', height=380)
        st.plotly_chart(fig_co2, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">ML Energy Forecast</div>', unsafe_allow_html=True)
    pred_country = st.selectbox("Country chunein", selected_countries, key='pred_c')
    pred_col = st.selectbox("Kaunsi metric?",
        ['primary_energy_consumption','renewables_share_energy','energy_per_capita'],
        format_func=lambda x: {'primary_energy_consumption':'⚡ Primary Energy (TWh)',
            'renewables_share_energy':'🌱 Renewables (%)',
            'energy_per_capita':'👤 Per Capita (kWh)'}[x])

    train_df = df[df['country']==pred_country][['year',pred_col]].dropna()

    if len(train_df) < 10:
        st.error("Is country ke liye data kam hai.")
    else:
        X = train_df[['year']].values
        y = train_df[pred_col].values
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, y)
        future_years = np.arange(train_df['year'].min(), pred_year+1).reshape(-1,1)
        preds = model.predict(future_years)
        pred_df = pd.DataFrame({'year': future_years.flatten(), 'predicted': preds})
        split_year = train_df['year'].max()

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=train_df['year'], y=train_df[pred_col],
            name='📊 Actual', line=dict(color='#38bdf8', width=2.5),
            mode='lines+markers', marker_size=4))
        fig_pred.add_trace(go.Scatter(
            x=pred_df[pred_df['year']>split_year]['year'],
            y=pred_df[pred_df['year']>split_year]['predicted'],
            name='🔮 Forecast', line=dict(color='#a78bfa', width=2.5, dash='dash'),
            mode='lines+markers', marker_size=4))
        fig_pred.add_vline(x=split_year, line_dash='dot', line_color='#64748b',
            annotation_text=f"Present ({split_year})",
            annotation_font_color='#64748b')
        fig_pred.update_layout(template=DARK, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,31,60,0.5)',
            title=f'{pred_country} — Forecast to {pred_year}',
            hovermode='x unified', height=430,
            legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_pred, use_container_width=True)

        final_pred = pred_df[pred_df['year']==pred_year]['predicted'].values[0]
        current_val = train_df[pred_col].iloc[-1]
        change_pct = ((final_pred - current_val) / current_val) * 100

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f"""<div class='pred-highlight'>
                <div class='pred-value'>{final_pred:,.1f}</div>
                <div class='pred-label'>Predicted {pred_year}</div>
            </div>""", unsafe_allow_html=True)
        with pc2:
            st.markdown(f"""<div class='pred-highlight'>
                <div class='pred-value'>{current_val:,.1f}</div>
                <div class='pred-label'>Current ({split_year})</div>
            </div>""", unsafe_allow_html=True)
        with pc3:
            arrow = "📈" if change_pct > 0 else "📉"
            st.markdown(f"""<div class='pred-highlight'>
                <div class='pred-value'>{arrow} {change_pct:+.1f}%</div>
                <div class='pred-label'>Expected Change</div>
            </div>""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.75rem; padding:10px 0;'>
⚡ Global Energy Intelligence · Our World in Data · Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
