import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from snowflake.snowpark import Session
from snowflake.cortex import complete, summarize
import json
from snowflake.core import Root

# Page configuration
st.set_page_config(
    page_title="Support Ticket Intelligence - Powered by Snowflake Cortex",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = {
        'data_source': 'AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS',
        'search_service': 'AI_SOL.SUPPORT.SUPPORT_SEARCH',
        'ai_model': 'claude-4-sonnet',
        'anomaly_threshold': 2.0,
        'date_range_days': 90  
    }

if 'tickets_data' not in st.session_state:
    st.session_state.tickets_data = None

if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None

if 'snowflake_session' not in st.session_state:
    st.session_state.snowflake_session = None

if 'ai_date_range' not in st.session_state:
    st.session_state.ai_date_range = (
        datetime.now().date() - timedelta(days=30),
        datetime.now().date()
    )

# Snowflake Connection
@st.cache_resource
def get_snowflake_session():
    """Initialize Snowflake session"""
    try:
        # Try to get active session first (if running in Snowflake)
        session = Session.builder.create()
        return session
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

# Configuration Section
def render_config_section():
    """Configuration panel for flexibility"""
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.expander("📊 Data Configuration", expanded=True):
        date_range = st.slider(
            "Analysis Period (days)",
            min_value=7,
            max_value=180,  
            value=90,       
            help="Number of days to analyze for Overview tab"
        )
    
    with st.sidebar.expander("🤖 AI Configuration", expanded=True):
        ai_model = st.selectbox(
            "Cortex AI Model",
            ["claude-4-sonnet", "claude-3-7-sonnet", "openai-gpt-4.1", "mistral-large2","llama3.1-70b"],
            help="Choose the AI model for analysis"
        )
        
        anomaly_threshold = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Lower values = more sensitive to anomalies"
        )
    
    
    # Save configuration (without data_source - that's now in top config)
    st.session_state.config.update({
        'ai_model': ai_model,
        'anomaly_threshold': anomaly_threshold,
        'date_range_days': date_range
    })
    
    # Connection status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Connection Status**")
    if st.session_state.snowflake_session:
        st.sidebar.success("✅ Snowflake Connected")
        st.sidebar.info("🤖 Cortex AI Ready")
    else:
        st.sidebar.error("❌ Snowflake Disconnected")


# Data Loading Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_support_tickets(_session, table_name, days_back=30):
    """Load support tickets from Snowflake table"""
    try:
        # First, get the column names from the table
        columns_query = f"SELECT * FROM {table_name} LIMIT 0"
        columns_df = _session.sql(columns_query).to_pandas()
        available_columns = columns_df.columns.tolist()
        
        # Determine date column to use
        date_column = None
        for col in ['SUBMIT_DATE', 'CREATED_DATE', 'TICKET_DATE', 'DATE']:
            if col in available_columns:
                date_column = col
                break
                
        if not date_column:
            raise ValueError("No suitable date column found in the table")
        
        # Calculate date filter
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Build dynamic column list based on what's available
        select_columns = []
        for col in ['TICKET_ID', date_column, 'CUSTOMER_ID', 'CUSTOMER_TIER', 
                   'CHANNEL', 'PRIORITY', 'STATUS', 'PRODUCT_AREA', 
                   'CLASSIFICATION', 'SENTIMENT', 'FIRST_RESPONSE_TIME_HOURS',
                   'RESOLUTION_TIME_HOURS', 'TICKET_SUBJECT', 
                   'TICKET_DESCRIPTION', 'AGENT_NOTES']:
            if col in available_columns:
                select_columns.append(col)
        
        # Query the table
        query = f"""
        SELECT {', '.join(select_columns)}
        FROM {table_name}
        WHERE {date_column} >= '{start_date}'
        ORDER BY {date_column} DESC
        """
        
        df = _session.sql(query).to_pandas()
        
        # Rename date column to SUBMIT_DATE if it's different
        if date_column != 'SUBMIT_DATE':
            df['SUBMIT_DATE'] = df[date_column]
        
        # Convert timestamp columns
        df['SUBMIT_DATE'] = pd.to_datetime(df['SUBMIT_DATE'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data from {table_name}: {str(e)}")
        return pd.DataFrame()

def detect_anomalies(df, threshold=2.0):
    """Detect anomalies in ticket volume and patterns using statistical analysis"""
    if df.empty:
        return {'volume_anomalies': pd.Series(dtype=float), 'category_anomalies': {}, 'z_scores': pd.Series(dtype=float)}
    
    # Daily ticket counts
    daily_counts = df.groupby(df['SUBMIT_DATE'].dt.date).size()
    
    if len(daily_counts) < 3:  # Need at least 3 days for meaningful analysis
        return {'volume_anomalies': pd.Series(dtype=float), 'category_anomalies': {}, 'z_scores': pd.Series(dtype=float)}
    
    # Calculate z-scores for anomaly detection
    mean_count = daily_counts.mean()
    std_count = daily_counts.std()
    
    if std_count == 0:  # All days have same count
        z_scores = pd.Series(0, index=daily_counts.index)
    else:
        z_scores = (daily_counts - mean_count) / std_count
    
    # Identify anomalous days
    anomalous_days = daily_counts[np.abs(z_scores) > threshold]
    
    # Category shifts
    daily_categories = df.groupby([df['SUBMIT_DATE'].dt.date, 'CLASSIFICATION']).size().unstack(fill_value=0)
    category_changes = {}
    
    for category in daily_categories.columns:
        cat_mean = daily_categories[category].mean()
        cat_std = daily_categories[category].std()
        if cat_std > 0:
            cat_z_scores = (daily_categories[category] - cat_mean) / cat_std
            category_changes[category] = daily_categories[category][np.abs(cat_z_scores) > threshold]
    
    return {
        'volume_anomalies': anomalous_days,
        'category_anomalies': category_changes,
        'z_scores': z_scores
    }

def run_cortex_analysis(_session, df, model="claude-3-sonnet"):
    """Run Cortex AI analysis on ticket data"""
    insights = []
    
    try:
        # Volume insights
        total_tickets = len(df)
        if total_tickets == 0:
            return [{'type': 'no_data', 'title': 'No Data Available', 'description': 'No tickets found in the selected time period.', 'severity': 'low'}]
        
        daily_avg = df.groupby(df['SUBMIT_DATE'].dt.date).size().mean()
        
        # Sentiment analysis using real data
        negative_pct = (df['SENTIMENT'] == 'Negative').sum() / total_tickets * 100
        if negative_pct > 40:
            insights.append({
                'type': 'sentiment',
                'title': 'Customer Sentiment Alert',
                'description': f'{negative_pct:.1f}% of recent tickets show negative sentiment, indicating potential customer satisfaction issues.',
                'severity': 'high' if negative_pct > 60 else 'medium'
            })
        
        # Category concentration analysis
        top_category = df['CLASSIFICATION'].value_counts().index[0]
        top_category_pct = (df['CLASSIFICATION'].value_counts().iloc[0] / total_tickets) * 100
        if top_category_pct > 50:
            insights.append({
                'type': 'category_concentration',
                'title': f'High Concentration of {top_category} Tickets',
                'description': f'{top_category} represents {top_category_pct:.1f}% of all tickets, suggesting a systematic issue.',
                'severity': 'medium'
            })
        
        # Response time analysis
        critical_tickets = df[df['PRIORITY'] == 'Critical']
        if len(critical_tickets) > 0:
            avg_critical_response = critical_tickets['FIRST_RESPONSE_TIME_HOURS'].mean()
            if avg_critical_response > 4:
                insights.append({
                    'type': 'response_time',
                    'title': 'Critical Ticket Response Delays',
                    'description': f'Critical tickets are taking an average of {avg_critical_response:.1f} hours for first response, exceeding typical SLA.',
                    'severity': 'high'
                })
        
        # Use Cortex AI for advanced analysis if we have ticket descriptions
        if 'TICKET_DESCRIPTION' in df.columns and not df['TICKET_DESCRIPTION'].isna().all():
            # AI-powered pattern detection using Cortex
            try:
                # Sample some ticket descriptions for analysis
                sample_descriptions = df['TICKET_DESCRIPTION'].dropna().head(10).tolist()
                if sample_descriptions:
                    # Use Cortex complete for pattern analysis
                    pattern_prompt = f"""
                    Analyze these customer support ticket descriptions and identify the top 3 most concerning patterns or trends:
                    
                    {chr(10).join(sample_descriptions[:5])}
                    
                    Provide insights in this format:
                    1. Pattern: [brief description]
                    2. Pattern: [brief description] 
                    3. Pattern: [brief description]
                    
                    Keep each pattern description under 50 words.
                    """
                    
                    pattern_analysis = complete(model, pattern_prompt)
                    
                    insights.append({
                        'type': 'ai_patterns',
                        'title': 'AI-Detected Issue Patterns',
                        'description': pattern_analysis,
                        'severity': 'medium'
                    })
            except Exception as e:
                st.warning(f"Cortex AI analysis temporarily unavailable: {str(e)}")
        
        # If no specific insights, provide general overview
        if not insights:
            insights.append({
                'type': 'overview',
                'title': 'System Operating Normally',
                'description': f'Analyzing {total_tickets} tickets with average daily volume of {daily_avg:.1f}. No significant anomalies detected.',
                'severity': 'low'
            })
        
        return insights
        
    except Exception as e:
        return [{
            'type': 'error',
            'title': 'Analysis Error',
            'description': f'Error running Cortex analysis: {str(e)}',
            'severity': 'medium'
        }]

def main():
    # Initialize Snowflake connection
    if st.session_state.snowflake_session is None:
        st.session_state.snowflake_session = get_snowflake_session()
    
    # Configuration sidebar
    render_config_section()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Support Ticket Intelligence</h1>
        <p>AI-Powered Customer Support Analytics with Snowflake Cortex</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check Snowflake connection
    if not st.session_state.snowflake_session:
        st.error("❌ Unable to connect to Snowflake. Please check your connection settings.")
        st.stop()
    
    # Load and process data for Overview (always use the configured date range)
    if st.session_state.tickets_data is None:
        with st.spinner("Loading support ticket data from Snowflake..."):
            df = load_support_tickets(
                st.session_state.snowflake_session,
                st.session_state.config['data_source'],
                st.session_state.config['date_range_days']
            )
            if not df.empty:
                st.session_state.tickets_data = df
                st.session_state.ai_insights = None  # Reset insights when data refreshes
            else:
                st.error("No data loaded. Please check your table name and permissions.")
                st.stop()
    
    df = st.session_state.tickets_data
    
    if df.empty:
        st.warning("No ticket data available for the selected time period.")
        st.stop()
    
    # Main tab structure
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 AI Insights", "🔍 Pattern Analysis", "🔎 Case Search"])
    
    with tab1:
        render_overview_tab(df)
    
    with tab2:
        render_ai_insights_tab(df)
    
    with tab3:
        render_pattern_analysis_tab(df)
    
    with tab4:
        render_case_search_tab()

def render_overview_tab(df):
    # Show data overview
    st.sidebar.info(f"📊 Loaded {len(df):,} tickets from {df['SUBMIT_DATE'].min().strftime('%Y-%m-%d')} to {df['SUBMIT_DATE'].max().strftime('%Y-%m-%d')}")
    
    # Detect anomalies
    anomalies = detect_anomalies(df, st.session_state.config['anomaly_threshold'])
    
    # Main dashboard content (streamlined for overview)
    render_main_dashboard(df, anomalies)
    
    # Real-time monitoring section
    st.markdown("---")
    st.subheader("⚡ Real-time Monitoring")
    
    # Current metrics
    today = datetime.now().date()
    today_tickets = df[df['SUBMIT_DATE'].dt.date == today]
    yesterday_tickets = df[df['SUBMIT_DATE'].dt.date == (today - timedelta(days=1))]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        today_count = len(today_tickets)
        yesterday_count = len(yesterday_tickets)
        delta = today_count - yesterday_count
        st.metric("Today's Volume", today_count, delta)
    
    with col2:
        if len(today_tickets) > 0:
            avg_response = today_tickets['FIRST_RESPONSE_TIME_HOURS'].mean()
            st.metric("Avg Response Time", f"{avg_response:.1f}h")
        else:
            st.metric("Avg Response Time", "0h")
    
    with col3:
        open_tickets = len(df[df['STATUS'].isin(['New', 'In Progress'])])
        st.metric("Open Tickets", open_tickets)
    
    # Recent ticket stream
    st.write("**Recent Ticket Stream**")
    if not df.empty:
        recent_tickets = df.head(10)[['TICKET_ID', 'CLASSIFICATION', 'PRIORITY', 'SENTIMENT', 'CHANNEL', 'SUBMIT_DATE']]
        st.dataframe(recent_tickets, use_container_width=True)

def render_ai_insights_tab(df):
    """Render the AI Insights tab"""
    st.subheader("🤖 AI Insights")
    
    # Date range selection for AI analysis
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.ai_date_range[0],
            max_value=datetime.now().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=st.session_state.ai_date_range[1],
            max_value=datetime.now().date()
        )
    
    with col3:
        # Data sampling toggle
        use_limited_data = st.toggle(
            "Limit data sample",
            value=False,
            help="Use only top 5 most recent tickets for analysis (faster) or full dataset"
        )
        
        # Generate Analysis button
        generate_analysis = st.button("🔄 Generate Analysis", type="primary")
    
    # Filter data for the selected date range
    filtered_df = df[
        (df['SUBMIT_DATE'].dt.date >= start_date) & 
        (df['SUBMIT_DATE'].dt.date <= end_date)
    ]
    
    if filtered_df.empty:
        st.warning("No tickets found in the selected date range.")
        return
    
    # Apply data sampling if toggle is enabled
    if use_limited_data:
        analysis_df = filtered_df.head(5)
        st.info(f"Analyzing top 5 most recent tickets from {start_date} to {end_date} (Limited sample)")
    else:
        analysis_df = filtered_df
        st.info(f"Analyzing {len(filtered_df):,} tickets from {start_date} to {end_date} (Full dataset)")
    
    # Generate AI insights only when button is pressed
    if generate_analysis:
        if start_date <= end_date:
            st.session_state.ai_date_range = (start_date, end_date)
            with st.spinner("Generating AI insights with Snowflake Cortex..."):
                st.session_state.ai_insights = run_cortex_analysis(
                    st.session_state.snowflake_session, 
                    analysis_df, 
                    st.session_state.config['ai_model']
                )
        else:
            st.error("Start date must be before end date")
            return
    
    # Processing status and results
    if st.session_state.ai_insights:
        with st.container():
            st.markdown(f"""
            <div class="insight-card">
                <h4>🧠 Cortex AI Analysis</h4>
                <p><strong>Model:</strong> {st.session_state.config['ai_model']}</p>
                <p><strong>Status:</strong> ✅ Analysis complete</p>
                <p><strong>Data Source:</strong> {st.session_state.config['data_source']}</p>
                <p><strong>Analysis Period:</strong> {start_date} to {end_date}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Show message when no analysis has been run
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b; border: 2px dashed #cbd5e1; border-radius: 12px;">
            <h3>🤖 Ready for AI Analysis</h3>
            <p>Select your date range and data sample preference above, then click <strong>"Generate Analysis"</strong> to get AI-powered insights.</p>
            <p><em>Features available:</em></p>
            <ul style="list-style: none; padding: 0;">
                <li>📊 Pattern detection and trend analysis</li>
                <li>🎯 Anomaly identification</li>
                <li>💭 Sentiment analysis insights</li>
                <li>⚡ Performance recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display insights
    if st.session_state.ai_insights:
        for insight in st.session_state.ai_insights:
            severity_color = {
                'high': '#e74c3c',
                'medium': '#f39c12',
                'low': '#2ecc71'
            }
            
            color = severity_color.get(insight['severity'], '#3498db')
            
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: {color};">
                <h4>{insight['title']}</h4>
                <p>{insight['description']}</p>
                <small><strong>Severity:</strong> {insight['severity'].upper()}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.session_state.ai_insights = None
            st.rerun()
    
    with col2:
        if st.session_state.ai_insights:
            report_data = {
                'total_tickets': len(filtered_df),
                'insights_count': len(st.session_state.ai_insights),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_source': st.session_state.config['data_source'],
                'date_range': {'start': str(start_date), 'end': str(end_date)},
                'insights': st.session_state.ai_insights
            }
            st.download_button(
                label="📊 Download Report",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"ai_insights_{start_date}_{end_date}.json",
                mime="application/json",
                use_container_width=True
            )

def render_pattern_analysis_tab(df):
    """Render the Pattern Analysis tab"""
    st.subheader("🔍 Deep Pattern Analysis")
    
    # Detect anomalies for this analysis
    anomalies = detect_anomalies(df, st.session_state.config['anomaly_threshold'])
    
    # Resolution Patterns Section
    st.markdown("##### 📊 Resolution Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Resolution time by priority
        if 'PRIORITY' in df.columns and 'RESOLUTION_TIME_HOURS' in df.columns:
            fig_priority_time = px.box(
                df,
                x='PRIORITY',
                y='RESOLUTION_TIME_HOURS',
                title='Resolution Time Distribution by Priority'
            )
            fig_priority_time.update_layout(height=350)
            st.plotly_chart(fig_priority_time, use_container_width=True)
        else:
            st.info("Priority or resolution time data not available")
    
    with col2:
        # Resolution time distribution
        if 'RESOLUTION_TIME_HOURS' in df.columns:
            fig_hist = px.histogram(
                df, 
                x='RESOLUTION_TIME_HOURS',
                nbins=20,
                title='Overall Resolution Time Distribution'
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Resolution time data not available")
    
    # Channel Performance Section
    st.markdown("##### 📞 Channel Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel volume
        if 'CHANNEL' in df.columns:
            channel_volume = df['CHANNEL'].value_counts()
            fig_channel_vol = px.bar(
                x=channel_volume.index,
                y=channel_volume.values,
                title='Ticket Volume by Channel',
                labels={'x': 'Channel', 'y': 'Number of Tickets'}
            )
            fig_channel_vol.update_layout(height=350)
            st.plotly_chart(fig_channel_vol, use_container_width=True)
        else:
            st.info("Channel data not available")
    
    with col2:
        # Channel sentiment
        if 'CHANNEL' in df.columns and 'SENTIMENT' in df.columns:
            channel_sentiment = pd.crosstab(df['CHANNEL'], df['SENTIMENT'], normalize='index') * 100
            
            if 'Positive' in channel_sentiment.columns:
                fig_sentiment = px.bar(
                    x=channel_sentiment.index,
                    y=channel_sentiment['Positive'],
                    title='Positive Sentiment by Channel (%)',
                    labels={'x': 'Channel', 'y': 'Positive Sentiment %'}
                )
                fig_sentiment.update_layout(height=350)
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("Positive sentiment data not available")
        else:
            st.info("Channel or sentiment data not available")
    
    
    # Sentiment Trends Section
    st.markdown("##### 💭 Sentiment Trends")
    if 'SENTIMENT' in df.columns:
        # Daily sentiment trends
        daily_sentiment = df.groupby([df['SUBMIT_DATE'].dt.date, 'SENTIMENT']).size().unstack(fill_value=0)
        
        fig_sentiment_trend = px.line(
            x=daily_sentiment.index,
            y=[daily_sentiment[col] for col in daily_sentiment.columns],
            title='Daily Sentiment Trends Over Time'
        )
        fig_sentiment_trend.update_layout(height=350)
        st.plotly_chart(fig_sentiment_trend, use_container_width=True)
    else:
        st.info("Sentiment data not available")

@st.cache_data
def search_support_tickets(prompt, limit=5):
    """Search support tickets using Cortex Search"""
    try:
        # Get the Snowflake session
        session = st.session_state.snowflake_session
        if not session:
            return None
        
        # Get search service from config
        search_service_parts = st.session_state.config['search_service'].split('.')
        database, schema, service = search_service_parts[0], search_service_parts[1], search_service_parts[2]
        
        # This defines the Search API location
        root = Root(session)
        svc = root.databases[database].schemas[schema].cortex_search_services[service]
        
        # Search with natural language query
        # Return the available columns: ticket_id, submit_date, customer_id, ticket_description
        resp = svc.search(
            query=prompt,
            columns=["ticket_id", "submit_date", "customer_id", "ticket_description"],
            limit=limit
        ).to_json()
        
        return resp
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def render_case_search_tab():
    """Render the Case Search tab"""
    st.subheader("🔎 Case Search")
    
    # Check if Snowflake session is available
    if not st.session_state.snowflake_session:
        st.error("❌ Snowflake connection required for search functionality")
        return
    
    # Search interface
    with st.form("ticket_search"):
        st.info('🤖 Ask a question in natural language to find the most relevant support tickets.')
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search for cases...",
                placeholder="Enter keywords, ticket ID, customer name, or issue description",
                help="Search through support tickets using natural language or specific terms"
            )
        
        with col2:
            result_limit = st.selectbox("# of results", [5, 10, 20, 50], index=0)
        
        search_button = st.form_submit_button("🔍 Search Cases", type="primary")
    
    # Search results container
    if search_button and search_query:
        st.markdown("---")
        
        with st.spinner("Searching support tickets..."):
            search_results = search_support_tickets(search_query, result_limit)
            search_summary_prompt = f"""
            Your task is to summarize the given support ticket search results based on the following question:
            <question> {search_query} </question>
            The ticket descriptions for you to summarize are below - be concise, under 100 words:
            <tickets>
            {search_results}
            </tickets>
            """
            try:
                summary = complete(st.session_state.config['ai_model'], search_summary_prompt)
            except Exception as e:
                st.warning(f"AI summary temporarily unavailable: {str(e)}")

        if search_results:
            # Summary results container
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <h4>📊 Search Summary</h4>
                    <p><strong>Query:</strong> {search_query}</p>
                    <p><strong>Results:</strong> Found {result_limit} most relevant cases</p>
                    <p><strong>AI Summary of Results:</strong> {summary}</p>
                </div>
                """, unsafe_allow_html=True)           
            
            
            # Parse and display individual results
            st.markdown("### 🎯 Top Matching Cases")
            
            try:
                import json
                json_conv = json.loads(search_results) if isinstance(search_results, str) else search_results
                search_df = pd.json_normalize(json_conv['results'])
                
                for i, (_, row) in enumerate(search_df.iterrows()):
                    with st.expander(f"📋 Case #{i+1} - {row['ticket_id']}", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Ticket ID:** {row['ticket_id']}  
                            **Customer ID:** {row['customer_id']}  
                            **Submit Date:** {datetime.strptime(row['submit_date'], '%Y-%m-%d').strftime('%b %d, %Y')}  
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **Matched Content:** Description  
                            **Data Source:** AI_SOL.SUPPORT  
                            """)
                        
                        st.markdown("**Ticket Description:**")
                        # Use Cortex Summarize for quick summary
                        try:
                            summary_text = summarize(row['ticket_description'], session=st.session_state.snowflake_session)
                            st.write("**AI Summary:** " + summary_text)
                        except Exception as e:
                            st.write(e)

                        # Show full description in a text area instead of nested expander
                        st.markdown("**Full Description:**")
                        st.text_area("", value=row['ticket_description'], height=100, disabled=True, key=f"desc_{i}", label_visibility="collapsed")
                        
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Error displaying search results: {str(e)}")
                st.text("Raw search results:")
                st.json(search_results)
        
        else:
            st.warning("No search results found. Please try a different search query.")
    
    elif search_button and not search_query:
        st.warning("Please enter a search query.")
    
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <h3>🔍 Search Through Support Cases</h3>
            <p>Enter a search query above to find relevant support tickets using AI-powered semantic search.</p>
            <p><em>Features available:</em></p>
            <ul style="list-style: none; padding: 0;">
                <li>🎯 Semantic search across ticket descriptions</li>
                <li>🤖 AI-powered result summaries</li>
                <li>📊 Search relevance scoring</li>
                <li>📝 Automatic ticket summarization</li>
                <li>🔗 Related case suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_main_dashboard(df, anomalies):
    """Main dashboard with key metrics and visualizations"""
    st.subheader("📊 Performance Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tickets = len(df)
        daily_avg = total_tickets / st.session_state.config['date_range_days']
        st.metric("Total Tickets", f"{total_tickets:,}", f"{daily_avg:.1f}/day")
    
    with col2:
        critical_count = len(df[df['PRIORITY'] == 'Critical'])
        critical_pct = (critical_count / total_tickets) * 100 if total_tickets > 0 else 0
        st.metric("Critical Issues", critical_count, f"{critical_pct:.1f}%")
    
    with col3:
        avg_resolution = df['RESOLUTION_TIME_HOURS'].mean()
        st.metric("Avg Resolution", f"{avg_resolution:.1f}h", "📈" if avg_resolution > 24 else "📉")
    
    with col4:
        negative_sentiment = (df['SENTIMENT'] == 'Negative').sum() / len(df) * 100 if len(df) > 0 else 0
        st.metric("Negative Sentiment", f"{negative_sentiment:.1f}%", "😔" if negative_sentiment > 40 else "😊")
    
    # Volume trend with anomaly highlighting
    st.subheader("📈 Ticket Volume Trend")
    
    daily_volume = df.groupby(df['SUBMIT_DATE'].dt.date).size().reset_index()
    daily_volume.columns = ['Date', 'Tickets']
    
    # Create volume chart
    fig_volume = px.line(daily_volume, x='Date', y='Tickets', 
                        title='Daily Ticket Volume with Anomaly Detection')
    
    # Highlight anomalies
    if len(anomalies['volume_anomalies']) > 0:
        anomaly_dates = anomalies['volume_anomalies'].index
        anomaly_volumes = anomalies['volume_anomalies'].values
        
        fig_volume.add_scatter(
            x=anomaly_dates,
            y=anomaly_volumes,
            mode='markers',
            marker=dict(color='red', size=12, symbol='diamond'),
            name='Volume Anomalies'
        )
    
    # Add mean line
    mean_volume = daily_volume['Tickets'].mean()
    fig_volume.add_hline(y=mean_volume, line_dash="dash", line_color="gray", 
                        annotation_text=f"Average: {mean_volume:.1f}")
    
    fig_volume.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Category and sentiment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Issue Categories")
        category_counts = df['CLASSIFICATION'].value_counts()
        fig_categories = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title='Ticket Distribution by Category'
        )
        fig_categories.update_layout(height=400)
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        st.subheader("💭 Sentiment Analysis")
        sentiment_counts = df['SENTIMENT'].value_counts()
        colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
        fig_sentiment = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title='Customer Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map=colors,
            labels={'x': 'Sentiment', 'y': 'Number of Tickets'}
        )
        fig_sentiment.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)



if __name__ == "__main__":
    main()
