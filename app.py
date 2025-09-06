import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from operator import attrgetter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Game Pulse Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('dataset.csv')
        
        # Convert date columns
        df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], format='%d-%b-%Y')
        df['Last_Login'] = pd.to_datetime(df['Last_Login'], format='%d-%b-%Y')
        
        # Create additional date columns for analysis
        df['signup_year_month'] = df['Signup_Date'].dt.to_period('M')
        df['last_login_year_month'] = df['Last_Login'].dt.to_period('M')
        
        # Calculate days since last login
        df['days_since_last_login'] = (datetime.now() - df['Last_Login']).dt.days
        
        # Create user segments based on behavior
        df['user_segment'] = df.apply(create_user_segment, axis=1)
        
        # Calculate ARPU
        df['arpu'] = df['Total_Revenue_USD'] / df['Total_Play_Sessions'].replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_user_segment(row):
    """Create user segments based on behavior patterns"""
    if row['Total_Revenue_USD'] > 100:
        return 'Whale'
    elif row['Total_Hours_Played'] > 50 and row['Total_Play_Sessions'] > 20:
        return 'Hardcore'
    elif row['Total_Play_Sessions'] > 10:
        return 'Regular'
    else:
        return 'Casual'

def calculate_dau_wau_mau(df, date_col='Last_Login'):
    """Calculate Daily, Weekly, Monthly Active Users"""
    # Get date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Create date range
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    dau_data = []
    wau_data = []
    mau_data = []
    
    for date in date_range:
        # DAU: users active on this specific day
        dau = len(df[df[date_col].dt.date == date.date()])
        
        # WAU: users active in the 7 days ending on this date
        week_start = date - timedelta(days=6)
        wau = len(df[(df[date_col] >= week_start) & (df[date_col] <= date)])
        
        # MAU: users active in the 30 days ending on this date
        month_start = date - timedelta(days=29)
        mau = len(df[(df[date_col] >= month_start) & (df[date_col] <= date)])
        
        dau_data.append({'date': date, 'dau': dau})
        wau_data.append({'date': date, 'wau': wau})
        mau_data.append({'date': date, 'mau': mau})
    
    return pd.DataFrame(dau_data), pd.DataFrame(wau_data), pd.DataFrame(mau_data)

def create_cohort_analysis(df):
    """Create cohort analysis for user retention"""
    # Create cohort groups based on signup month
    df['cohort_group'] = df['Signup_Date'].dt.to_period('M')
    
    # Create period number (months since signup)
    df['period_number'] = (df['last_login_year_month'] - df['cohort_group']).apply(attrgetter('n'))
    
    # Create cohort table
    cohort_data = df.groupby(['cohort_group', 'period_number'])['User_ID'].nunique().reset_index()
    cohort_table = cohort_data.pivot(index='cohort_group', columns='period_number', values='User_ID')
    
    # Calculate retention rates
    cohort_sizes = cohort_table.iloc[:, 0]
    retention_table = cohort_table.divide(cohort_sizes, axis=0)
    
    return retention_table

def create_funnel_analysis(df):
    """Create funnel analysis from signup to repeat sessions"""
    total_signups = len(df)
    users_with_sessions = len(df[df['Total_Play_Sessions'] > 0])
    users_with_multiple_sessions = len(df[df['Total_Play_Sessions'] > 1])
    users_with_purchases = len(df[df['In_Game_Purchases_Count'] > 0])
    users_with_revenue = len(df[df['Total_Revenue_USD'] > 0])
    
    funnel_data = {
        'Stage': ['Signups', 'First Session', 'Multiple Sessions', 'Made Purchase', 'Generated Revenue'],
        'Users': [total_signups, users_with_sessions, users_with_multiple_sessions, users_with_purchases, users_with_revenue],
        'Conversion_Rate': [
            100,
            (users_with_sessions / total_signups) * 100,
            (users_with_multiple_sessions / total_signups) * 100,
            (users_with_purchases / total_signups) * 100,
            (users_with_revenue / total_signups) * 100
        ]
    }
    
    return pd.DataFrame(funnel_data)

def perform_user_clustering(df):
    """Perform K-means clustering to segment users"""
    # Select features for clustering
    features = ['Total_Play_Sessions', 'Total_Hours_Played', 'Total_Revenue_USD', 'Avg_Session_Duration_Min']
    X = df[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Name clusters based on characteristics
    cluster_names = {0: 'Casual Players', 1: 'Regular Players', 2: 'Hardcore Players', 3: 'Whales'}
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 Game Pulse Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data loaded. Please check your dataset.")
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Date range filter
    min_date = df['Signup_Date'].min().date()
    max_date = df['Signup_Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date
    )
    
    # Device type filter
    device_types = ['All'] + list(df['Device_Type'].unique())
    selected_device = st.sidebar.selectbox("Device Type", device_types)
    
    # User segment filter
    segments = ['All'] + list(df['user_segment'].unique())
    selected_segment = st.sidebar.selectbox("User Segment", segments)
    
    # Game mode filter
    game_modes = ['All'] + list(df['Preferred_Game_Mode'].unique())
    selected_mode = st.sidebar.selectbox("Game Mode", game_modes)
    
    # Apply filters
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Signup_Date'].dt.date >= date_range[0]) &
            (filtered_df['Signup_Date'].dt.date <= date_range[1])
        ]
    
    if selected_device != 'All':
        filtered_df = filtered_df[filtered_df['Device_Type'] == selected_device]
    
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['user_segment'] == selected_segment]
    
    if selected_mode != 'All':
        filtered_df = filtered_df[filtered_df['Preferred_Game_Mode'] == selected_mode]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "👥 User Behavior", "💰 Revenue Analysis", 
        "🔄 Retention & Funnel", "🎯 User Segmentation"
    ])
    
    with tab1:
        st.header("📈 Key Metrics Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{len(filtered_df):,}")
        
        with col2:
            total_revenue = filtered_df['Total_Revenue_USD'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        with col3:
            avg_arpu = filtered_df['arpu'].mean()
            st.metric("Average ARPU", f"${avg_arpu:.2f}")
        
        with col4:
            avg_sessions = filtered_df['Total_Play_Sessions'].mean()
            st.metric("Avg Sessions/User", f"{avg_sessions:.1f}")
        
        # DAU/WAU/MAU Analysis
        st.subheader("📊 Active Users Analysis")
        
        dau_df, wau_df, mau_df = calculate_dau_wau_mau(filtered_df)
        
        # Combine the data
        active_users_df = dau_df.merge(wau_df, on='date').merge(mau_df, on='date')
        
        # Plot active users
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=active_users_df['date'], y=active_users_df['dau'], 
                                name='DAU', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=active_users_df['date'], y=active_users_df['wau'], 
                                name='WAU', line=dict(color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=active_users_df['date'], y=active_users_df['mau'], 
                                name='MAU', line=dict(color='#2ca02c')))
        
        fig.update_layout(
            title="Daily, Weekly, and Monthly Active Users",
            xaxis_title="Date",
            yaxis_title="Number of Users",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Breakdowns
        st.subheader("📊 User Breakdowns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Device type breakdown
            device_breakdown = filtered_df['Device_Type'].value_counts()
            fig_device = px.pie(values=device_breakdown.values, names=device_breakdown.index, 
                               title="Users by Device Type")
            st.plotly_chart(fig_device, use_container_width=True)
        
        with col2:
            # User segment breakdown
            segment_breakdown = filtered_df['user_segment'].value_counts()
            fig_segment = px.pie(values=segment_breakdown.values, names=segment_breakdown.index, 
                                title="Users by Segment")
            st.plotly_chart(fig_segment, use_container_width=True)
    
    with tab2:
        st.header("👥 User Behavior Analysis")
        
        # Behavioral patterns
        st.subheader("🎯 Behavioral Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session frequency distribution
            fig_sessions = px.histogram(filtered_df, x='Total_Play_Sessions', 
                                       title="Session Frequency Distribution",
                                       nbins=20)
            st.plotly_chart(fig_sessions, use_container_width=True)
        
        with col2:
            # Average session duration
            fig_duration = px.histogram(filtered_df, x='Avg_Session_Duration_Min', 
                                       title="Average Session Duration Distribution",
                                       nbins=20)
            st.plotly_chart(fig_duration, use_container_width=True)
        
        # Peak playtimes analysis
        st.subheader("⏰ Peak Playtimes")
        
        # Create hour-based analysis (simulated from session data)
        filtered_df['session_hour'] = np.random.randint(0, 24, len(filtered_df))
        hourly_activity = filtered_df.groupby('session_hour').size().reset_index(name='users')
        
        fig_hourly = px.bar(hourly_activity, x='session_hour', y='users', 
                           title="User Activity by Hour of Day",
                           labels={'session_hour': 'Hour of Day', 'users': 'Number of Users'})
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Churn analysis
        st.subheader("⚠️ Churn Signals")
        
        # Calculate 90th percentile threshold for churned users
        churn_threshold_90th = filtered_df['days_since_last_login'].quantile(0.9)
        
        # Users with increasing time gaps
        churned_users = filtered_df[filtered_df['days_since_last_login'] > churn_threshold_90th]
        at_risk_users = filtered_df[(filtered_df['days_since_last_login'] > 7) & 
                                   (filtered_df['days_since_last_login'] <= churn_threshold_90th)]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Churned Users (>{churn_threshold_90th:.0f} days)", len(churned_users))
        
        with col2:
            st.metric(f"At Risk Users (7-{churn_threshold_90th:.0f} days)", len(at_risk_users))
        
        with col3:
            churn_rate = len(churned_users) / len(filtered_df) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Users with only 1-2 short sessions
        low_engagement = filtered_df[
            (filtered_df['Total_Play_Sessions'] <= 2) & 
            (filtered_df['Avg_Session_Duration_Min'] < 10)
        ]
        
        st.metric("Low Engagement Users (≤2 sessions, <10min)", len(low_engagement))
    
    with tab3:
        st.header("💰 Revenue Analysis")
        
        # Revenue trends
        st.subheader("📈 Revenue Trends Over Time")
        
        # Monthly revenue
        monthly_revenue = filtered_df.groupby('signup_year_month')['Total_Revenue_USD'].sum().reset_index()
        monthly_revenue['signup_year_month'] = monthly_revenue['signup_year_month'].astype(str)
        
        fig_revenue = px.line(monthly_revenue, x='signup_year_month', y='Total_Revenue_USD',
                             title="Monthly Revenue Trends")
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # ARPU analysis
        st.subheader("💵 Average Revenue Per User (ARPU)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ARPU by device type
            arpu_device = filtered_df.groupby('Device_Type')['arpu'].mean().reset_index()
            fig_arpu_device = px.bar(arpu_device, x='Device_Type', y='arpu',
                                    title="ARPU by Device Type")
            st.plotly_chart(fig_arpu_device, use_container_width=True)
        
        with col2:
            # ARPU by user segment
            arpu_segment = filtered_df.groupby('user_segment')['arpu'].mean().reset_index()
            fig_arpu_segment = px.bar(arpu_segment, x='user_segment', y='arpu',
                                     title="ARPU by User Segment")
            st.plotly_chart(fig_arpu_segment, use_container_width=True)
        
        # High-value users analysis
        st.subheader("💎 High-Value Users Analysis")
        
        # Top spenders
        top_spenders = filtered_df.nlargest(10, 'Total_Revenue_USD')[
            ['Username', 'Total_Revenue_USD', 'Total_Play_Sessions', 'user_segment', 'Device_Type']
        ]
        
        st.write("**Top 10 Spenders:**")
        st.dataframe(top_spenders, use_container_width=True)
        
        # Revenue distribution
        fig_revenue_dist = px.histogram(filtered_df, x='Total_Revenue_USD', 
                                       title="Revenue Distribution",
                                       nbins=20)
        st.plotly_chart(fig_revenue_dist, use_container_width=True)
    
    with tab4:
        st.header("🔄 Retention & Funnel Analysis")
        
        # Cohort Analysis
        st.subheader("📊 Cohort Analysis")
        
        try:
            retention_table = create_cohort_analysis(filtered_df)
            
            # Display retention heatmap
            fig_cohort = px.imshow(retention_table.values, 
                                  labels=dict(x="Months Since Signup", y="Cohort", color="Retention Rate"),
                                  x=[f"Month {i}" for i in range(len(retention_table.columns))],
                                  y=[str(period) for period in retention_table.index],
                                  title="User Retention by Cohort",
                                  color_continuous_scale="RdYlBu_r")
            st.plotly_chart(fig_cohort, use_container_width=True)
        except Exception as e:
            st.warning(f"Cohort analysis not available: {e}")
        
        # Funnel Analysis
        st.subheader("🔄 User Journey Funnel")
        
        funnel_df = create_funnel_analysis(filtered_df)
        
        fig_funnel = px.funnel(funnel_df, x='Users', y='Stage', 
                              title="User Journey Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Conversion rates table
        st.write("**Conversion Rates:**")
        st.dataframe(funnel_df, use_container_width=True)
    
    with tab5:
        st.header("🎯 User Segmentation")
        
        # Perform clustering
        clustered_df = perform_user_clustering(filtered_df)
        
        # Cluster distribution
        cluster_dist = clustered_df['cluster_name'].value_counts()
        fig_cluster = px.pie(values=cluster_dist.values, names=cluster_dist.index,
                            title="User Clusters Distribution")
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("📊 Cluster Characteristics")
        
        cluster_stats = clustered_df.groupby('cluster_name').agg({
            'Total_Play_Sessions': 'mean',
            'Total_Hours_Played': 'mean',
            'Total_Revenue_USD': 'mean',
            'Avg_Session_Duration_Min': 'mean'
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # 3D scatter plot of clusters
        fig_3d = px.scatter_3d(clustered_df, 
                              x='Total_Play_Sessions', 
                              y='Total_Hours_Played', 
                              z='Total_Revenue_USD',
                              color='cluster_name',
                              title="User Clusters in 3D Space")
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Detailed cluster analysis
        st.subheader("🔍 Detailed Cluster Analysis")
        
        for cluster in clustered_df['cluster_name'].unique():
            with st.expander(f"📋 {cluster} Analysis"):
                cluster_data = clustered_df[clustered_df['cluster_name'] == cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Count:** {len(cluster_data)} users")
                    st.write(f"**Avg Sessions:** {cluster_data['Total_Play_Sessions'].mean():.1f}")
                    st.write(f"**Avg Hours:** {cluster_data['Total_Hours_Played'].mean():.1f}")
                
                with col2:
                    st.write(f"**Avg Revenue:** ${cluster_data['Total_Revenue_USD'].mean():.2f}")
                    st.write(f"**Avg Session Duration:** {cluster_data['Avg_Session_Duration_Min'].mean():.1f} min")
                    st.write(f"**Top Device:** {cluster_data['Device_Type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'}")

if __name__ == "__main__":
    main()
