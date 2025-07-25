import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, count_distinct, max, ai_agg, concat, lit, ai_filter
from snowflake.cortex import complete
from snowflake.core import Root

# Initialize Snowflake session
session = get_active_session()

# App configuration
st.set_page_config(
    page_title="Snowball Analytics - AI Sales Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Date Configuration for Demo (remove comment to use today's date)
current_date = pd.Timestamp(2025, 7, 23)
#current_date = pd.Timestamp.now()


# Load external CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sales data configuration
TASKS_TABLE = 'AI_SOL.SALES.TASKS'
ACCOUNTS_TABLE = 'AI_SOL.SALES.ACCOUNTS' 
OPPORTUNITIES_TABLE = 'AI_SOL.SALES.OPPORTUNITIES'
CONTACTS_TABLE = 'AI_SOL.SALES.CONTACTS'

# Modern header
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">🚀 Snowball Analytics</h1>
    <p class="main-subtitle">AI Sales Intelligence Platform • Powered by Snowflake Cortex</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def get_sales_overview():
    """Get high-level sales metrics"""
    return session.sql(f"""
        SELECT 
            'Tasks' as data_type,
            COUNT(*) as record_count,
            MIN(activity_date) as earliest_date,
            MAX(activity_date) as latest_date
        FROM {TASKS_TABLE}
        UNION ALL
        SELECT 
            'Opportunities' as data_type,
            COUNT(*) as record_count,
            MIN(close_date) as earliest_date,
            MAX(close_date) as latest_date
        FROM {OPPORTUNITIES_TABLE}
        UNION ALL
        SELECT 
            'Accounts' as data_type,
            COUNT(*) as record_count,
            MIN(created_date) as earliest_date,
            MAX(last_modified_date) as latest_date
        FROM {ACCOUNTS_TABLE}
    """).to_pandas()

@st.cache_data
def get_opportunities_data():
    """Get opportunity data for dashboard"""
    return session.sql(f"""
        SELECT 
            o.STAGE_NAME,
            o.CLOSE_DATE,
            o.AMOUNT,
            o.PROBABILITY,
            a.BILLING_COUNTRY as GEO,
            a.ANNUAL_REVENUE,
            a.NAME as ACCOUNT_NAME,
            o.LEAD_SOURCE,
            o.NAME as OPPORTUNITY_NAME
        FROM {OPPORTUNITIES_TABLE} o
        JOIN {ACCOUNTS_TABLE} a ON o.ACCOUNT_ID = a.ID
        WHERE o.CLOSE_DATE >= '2024-01-01'
    """).to_pandas()

@st.cache_data
def get_meeting_brief_data(account_name):
    """Get comprehensive account intelligence for meeting preparation"""
    meeting_query = f"""
        SELECT 
            a.name as account_name,
            a.industry,
            a.annual_revenue,
            COUNT(DISTINCT t.id) as total_activities,
            COUNT(DISTINCT o.id) as total_opportunities,
            MAX(t.activity_date) as last_activity_date
        FROM {ACCOUNTS_TABLE} a
        LEFT JOIN {TASKS_TABLE} t ON a.id = t.account_id
        LEFT JOIN {OPPORTUNITIES_TABLE} o ON a.id = o.account_id
        WHERE a.name = '{account_name}'
        GROUP BY a.name, a.industry, a.annual_revenue
    """
    return session.sql(meeting_query).to_pandas()

@st.cache_data
def get_account_contacts(account_name):
    """Get contacts for a specific account"""
    contacts_query = f"""
        SELECT 
            c.first_name || ' ' || c.last_name as contact_name,
            c.title,
            c.department,
            c.email
        FROM {CONTACTS_TABLE} c
        JOIN {ACCOUNTS_TABLE} a ON c.account_id = a.id
        WHERE a.name = '{account_name}'
    """
    return session.sql(contacts_query).to_pandas()

# Tab-based navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🤖 AI Assistant", 
    "🤝 Meeting Prep", 
    "🎯 MEDDIC Analysis", 
    "✉️ Outreach Generator"
])

# ====================
# SALES DASHBOARD TAB
# ====================
with tab1:

    
    # Get and process data
    opps_df = get_opportunities_data()
    opps_df['CLOSE_DATE'] = pd.to_datetime(opps_df['CLOSE_DATE'])
    opps_df['WEIGHTED_AMOUNT'] = opps_df['AMOUNT'] * (opps_df['PROBABILITY'] / 100)
    
    # Calculate insights
    opps_df['DAYS_TO_CLOSE'] = (opps_df['CLOSE_DATE'] - current_date).dt.days
    opps_df['IS_OVERDUE'] = opps_df['DAYS_TO_CLOSE'] < 0
    opps_df['CLOSING_NEXT_30'] = opps_df['DAYS_TO_CLOSE'].between(0, 30)
    
    # Streamlined filters - focus on the most important
    st.markdown('<div class="section-header">🎯 Pipeline Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        geo_options = ['All Regions'] + sorted(opps_df['GEO'].unique().tolist())
        selected_geo = st.selectbox('Territory', options=geo_options, index=0)
    
    with col2:
        time_filter = st.selectbox(
            'Time Focus', 
            ['All Time', 'Next 30 Days', 'This Quarter', 'Overdue Only']
        )
    
    with col3:
        min_deal_size = st.selectbox(
            'Min Deal Size', 
            [0, 50000, 100000, 250000],
            format_func=lambda x: f"€{x:,}+" if x > 0 else "All Deals"
        )
    
    # Apply filters
    filtered_df = opps_df.copy()
    
    if selected_geo != 'All Regions':
        filtered_df = filtered_df[filtered_df['GEO'] == selected_geo]
    
    if time_filter == 'Next 30 Days':
        filtered_df = filtered_df[filtered_df['CLOSING_NEXT_30']]
    elif time_filter == 'This Quarter':
        quarter_start = pd.Timestamp(current_date.year, ((current_date.month - 1) // 3) * 3 + 1, 1)
        quarter_end = quarter_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        filtered_df = filtered_df[(filtered_df['CLOSE_DATE'] >= quarter_start) & (filtered_df['CLOSE_DATE'] <= quarter_end)]
    elif time_filter == 'Overdue Only':
        filtered_df = filtered_df[filtered_df['IS_OVERDUE']]
    
    filtered_df = filtered_df[filtered_df['AMOUNT'] >= min_deal_size]
    
    # Key Metrics - Clean and focused
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pipeline = filtered_df['AMOUNT'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">€{total_pipeline:,.0f}</div>
            <div class="metric-label">Total Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        weighted_pipeline = filtered_df['WEIGHTED_AMOUNT'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">€{weighted_pipeline:,.0f}</div>
            <div class="metric-label">Weighted Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        deal_count = len(filtered_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{deal_count}</div>
            <div class="metric-label">Active Deals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_deal_size = filtered_df['AMOUNT'].mean() if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">€{avg_deal_size:,.0f}</div>
            <div class="metric-label">Avg Deal Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Single Priority Alert - Most Important Action
    open_deals = filtered_df[~filtered_df['STAGE_NAME'].isin(['Closed Won', 'Closed Lost'])]
    overdue_deals = open_deals[open_deals['IS_OVERDUE']]
    closing_soon = open_deals[open_deals['CLOSING_NEXT_30']]
    
    if len(overdue_deals) > 0:
        priority_message = f"🚨 **{len(overdue_deals)} deals are overdue** (€{overdue_deals['AMOUNT'].sum():,.0f}) - Immediate action required"
    elif len(closing_soon) > 0:
        priority_message = f"⏰ **{len(closing_soon)} deals closing in 30 days** (€{closing_soon['AMOUNT'].sum():,.0f}) - Focus on closing activities"
    else:
        high_value = open_deals[open_deals['AMOUNT'] >= 100000]
        priority_message = f"💎 **{len(high_value)} high-value opportunities** (€{high_value['AMOUNT'].sum():,.0f}) - Maintain momentum"
    
    st.markdown(f"""
    <div class="priority-alert">
        <div class="priority-title">Priority Action Required</div>
        <div class="priority-content">{priority_message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Analytics - Side by side for better space usage
    st.markdown('<div class="section-header">📊 Pipeline Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pipeline by Stage - Cleaner chart
        stage_analysis = filtered_df.groupby('STAGE_NAME').agg({
            'AMOUNT': 'sum',
            'WEIGHTED_AMOUNT': 'sum'
        }).reset_index()
        
        fig_pipeline = px.bar(
            stage_analysis,
            x='STAGE_NAME',
            y=['AMOUNT', 'WEIGHTED_AMOUNT'],
            title='Pipeline by Stage',
            labels={'value': 'Amount (€)', 'STAGE_NAME': 'Stage'},
            color_discrete_sequence=['#e3f2fd', '#1976d2'],
            barmode='group'
        )
        fig_pipeline.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pipeline, use_container_width=True)
    
    with col2:
        # Top Accounts - More actionable view
        top_accounts = filtered_df.groupby('ACCOUNT_NAME')['WEIGHTED_AMOUNT'].sum().reset_index()
        top_accounts = top_accounts.sort_values('WEIGHTED_AMOUNT', ascending=True).tail(8)
        
        fig_accounts = px.bar(
            top_accounts,
            x='WEIGHTED_AMOUNT',
            y='ACCOUNT_NAME',
            orientation='h',
            title='Top Accounts by Weighted Pipeline',
            labels={'WEIGHTED_AMOUNT': 'Weighted Amount (€)', 'ACCOUNT_NAME': 'Account'},
            color='WEIGHTED_AMOUNT',
            color_continuous_scale='Blues'
        )
        fig_accounts.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_accounts, use_container_width=True)
    
    # AI-Powered Insights Section
    st.markdown('<div class="section-header">🤖 AI Insights & Recommendations</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Generate AI Insights", key="generate_insights"):
        with st.spinner("Analyzing pipeline data with AI..."):
            try:
                # Use AI to analyze pipeline trends and generate insights
                pipeline_summary = filtered_df.groupby('STAGE_NAME').agg({
                    'AMOUNT': ['count', 'sum', 'mean'],
                    'PROBABILITY': 'mean',
                    'DAYS_TO_CLOSE': 'mean'
                }).round(2)
                
                # Get AI insights
                insights_prompt = f"""
                Analyze this sales pipeline data and provide 3 specific, actionable insights:
                
                Pipeline Summary:
                - Total Pipeline: €{total_pipeline:,.0f}
                - Weighted Pipeline: €{weighted_pipeline:,.0f} 
                - Active Deals: {deal_count}
                - Average Deal Size: €{avg_deal_size:,.0f}
                - Overdue Deals: {len(overdue_deals)}
                - Closing in 30 Days: {len(closing_soon)}
                
                Stage Distribution:
                {pipeline_summary.to_string()}
                
                Provide insights in this format:
                1. **Trend Analysis**: [One sentence about pipeline health/trends]
                2. **Risk Assessment**: [One sentence about risks or concerns]  
                3. **Recommendation**: [One specific action to take]
                
                Keep each insight to one clear, actionable sentence. Focus on what the sales team should do next.
                """
                
                ai_insights = complete('claude-4-sonnet', insights_prompt)
                
                # Display AI insights in clean format
                insights_lines = ai_insights.strip().split('\n')
                for line in insights_lines:
                    if line.strip() and ('**' in line or line.startswith(('1.', '2.', '3.'))):
                        # Extract icon based on content
                        if 'Trend' in line or '1.' in line:
                            icon = "📈"
                        elif 'Risk' in line or '2.' in line:
                            icon = "⚠️"
                        elif 'Recommendation' in line or '3.' in line:
                            icon = "💡"
                        else:
                            icon = "🎯"
                        
                        # Clean up the line
                        clean_line = line.replace('**', '').replace('1. ', '').replace('2. ', '').replace('3. ', '')
                        
                        st.markdown(f"""
                        <div class="ai-insight">
                            <div class="insight-header">
                                <span class="insight-icon">{icon}</span>
                                <div class="insight-title">AI Insight</div>
                            </div>
                            <div class="insight-text">{clean_line}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    else:
        # Show placeholder insights
        st.markdown("""
        <div class="ai-insight">
            <div class="insight-header">
                <span class="insight-icon">🤖</span>
                <div class="insight-title">AI Analysis Ready</div>
            </div>
            <div class="insight-text">Click "Generate AI Insights" above to get personalized recommendations based on your current pipeline data.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Actions - Only show if relevant deals exist
    if len(overdue_deals) > 0 or len(closing_soon) > 0:
        with st.expander("📋 Priority Deals Requiring Action", expanded=False):
            priority_deals = pd.concat([
                overdue_deals.assign(Priority='Overdue'),
                closing_soon[~closing_soon.index.isin(overdue_deals.index)].assign(Priority='Closing Soon')
            ]).sort_values(['Priority', 'AMOUNT'], ascending=[True, False])
            
            if len(priority_deals) > 0:
                display_cols = ['OPPORTUNITY_NAME', 'ACCOUNT_NAME', 'STAGE_NAME', 'AMOUNT', 'CLOSE_DATE', 'Priority']
                st.dataframe(
                    priority_deals[display_cols].head(10),
                    use_container_width=True,
                    hide_index=True
                ) 

# =========================
# CONVERSATIONAL ASSISTANT TAB
# =========================
with tab2:
    st.info("Ask natural language questions about accounts, opportunities, and sales activities with this RAG-style assistant.")
    

    
    st.markdown("### 💡 Try asking questions like:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="question-card">
            <span class="question-icon">🏢</span>
            <span class="question-text">"What's the latest activity with Hamburg Digital?"</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="question-card">
            <span class="question-icon">💰</span>
            <span class="question-text">"Which accounts have budget approval discussions?"</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="question-card">
            <span class="question-icon">🏆</span>
            <span class="question-text">"Show me deals where competitors were mentioned"</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="question-card">
            <span class="question-icon">🔧</span>
            <span class="question-text">"Find me occaisions where we've had technical deep dives"</span>
        </div>
        """, unsafe_allow_html=True)
    
    root = Root(session)
    
    # Conversational interface
    with st.form("sales_query"):
        user_question = st.text_input(
            "Ask your sales question:",
            placeholder="What's the latest activity with Hamburg Digital?"
        )
        
        search_submitted = st.form_submit_button("Search")
    
    if search_submitted and user_question:
        with st.spinner("Analyzing your question..."):
            try:
                # Search sales data using Cortex Search Service
                search_service = (root
                    .databases["ai_sol"]
                    .schemas["sales"]
                    .cortex_search_services["sales_search"]
                )
                
                search_results = search_service.search(
                    query=user_question,
                    columns=["id", "description", "account_name", 
                            "type", "activity_date"],
                    limit=10
                )
                
                # Generate AI response based on search results
                model = 'claude-4-sonnet'
                
                context_prompt = f"""
                Based on the following sales data, answer this question: "{user_question}"
                
                Sales Data:
                {search_results.to_str()}
                
                Provide a helpful, contextual answer that summarizes the relevant information and suggests next steps if appropriate. 
                Be specific about accounts, dates, and activities mentioned.
                """
                
                ai_response = complete(model, context_prompt)
                

                
                # Display AI Response in styled card
                st.markdown(f"""
                <div class="ai-response-card">
                    <div class="ai-response-header">
                        <span class="ai-response-icon">🎯</span>
                        <div class="ai-response-title">AI Analysis & Insights</div>
                    </div>
                    <div class="ai-response-text">{ai_response}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display search results
                st.warning(""" 📋 Supporting Sales Activities """)

                results_json_string = search_results.to_json()  
                results_data = json.loads(results_json_string)  
                
                if results_data and 'results' in results_data:
                    for i, result in enumerate(results_data['results'][:3]):  # Show top 3
                        account_name = result.get('account_name', 'Unknown Account')
                        activity_type = result.get('type', 'Activity')
                        activity_date = result.get('activity_date', 'N/A')
                        description = result.get('description', 'No description available')
                        
                        with st.expander(f"📌 {account_name} - {activity_type}"):
                            st.markdown(f"""
                            <div class="activity-item">
                                <div class="activity-detail"><strong>Date:</strong> {activity_date}</div>
                                <div class="activity-detail"><strong>Details:</strong> {description}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================
# MEETING PREPARATION TAB
# ====================
with tab3:
    st.info("Generate AI-powered meeting briefs based on specific opportunities and account context.")
    
    # Get accounts list
    accounts_df = session.table(ACCOUNTS_TABLE)
    accounts_list = accounts_df.select('NAME').distinct().sort('NAME').to_pandas()['NAME'].tolist()
    
    selected_account = st.selectbox(
        "Select Account for Meeting Prep:",
        options=accounts_list,
        index=accounts_list.index('Hamburg Digital') if 'Hamburg Digital' in accounts_list else 0,
        key="meeting_account_selector"
    )
    
    # Get opportunities for selected account
    opportunities_query = f"""
        SELECT 
            o.name as opportunity_name,
            o.stage_name,
            o.amount,
            o.close_date,
            o.id
        FROM {OPPORTUNITIES_TABLE} o
        JOIN {ACCOUNTS_TABLE} a ON o.account_id = a.id
        WHERE a.name = '{selected_account.replace("'", "''")}'
        AND o.stage_name NOT IN ('Closed Won', 'Closed Lost')
        ORDER BY o.amount DESC, o.close_date ASC
    """
    
    opportunities_df = session.sql(opportunities_query).to_pandas()
    
    if not opportunities_df.empty:
        # Create display options for opportunities
        opp_options = []
        for _, opp in opportunities_df.iterrows():
            display_name = f"{opp['OPPORTUNITY_NAME']} - {opp['STAGE_NAME']} (€{opp['AMOUNT']:,.0f})"
            opp_options.append(display_name)
        
        selected_opp_display = st.selectbox(
            "Select Opportunity for Meeting Focus:",
            options=opp_options,
            key="meeting_opp_selector"
        )
        
        # Get the selected opportunity details
        selected_opp_index = opp_options.index(selected_opp_display)
        selected_opportunity = opportunities_df.iloc[selected_opp_index]
        
        if st.button("📋 Generate Meeting Brief"):
            with st.spinner("Generating opportunity-focused meeting brief..."):
                try:
                    # Get account-level metrics
                    account_data = get_meeting_brief_data(selected_account)
                    
                    # Get opportunity-specific meeting brief
                    opp_meeting_query = f"""
                        SELECT 
                            o.name as opportunity_name,
                            o.stage_name,
                            o.amount,
                            o.close_date,
                            o.probability,
                            o.lead_source,
                            AI_AGG(
                                t.description,
                                'Create a focused meeting preparation brief for this specific opportunity. Include: 1) Opportunity status and next steps 2) Key stakeholders involved in this deal 3) Technical requirements and business drivers 4) Competitive situation and positioning 5) Risk factors and mitigation strategies 6) Specific talking points and questions for advancing this opportunity. Format as opportunity-focused executive brief.'
                            ) as opportunity_brief
                        FROM {OPPORTUNITIES_TABLE} o
                        LEFT JOIN {TASKS_TABLE} t ON (o.id = t.what_id)
                        WHERE o.name = '{selected_opportunity['OPPORTUNITY_NAME'].replace("'", "''")}'
                        GROUP BY ALL
                    """
                    
                    opp_meeting_data = session.sql(opp_meeting_query).to_pandas()
                    
                    if not opp_meeting_data.empty and not account_data.empty:
                        opp_brief = opp_meeting_data.iloc[0]
                        account_brief = account_data.iloc[0]
                        
                        st.success(f"Meeting Brief Generated for {selected_opportunity['OPPORTUNITY_NAME']}")
                        
                        # Display account-level metrics (unchanged)
                        st.markdown("### 🏢 Account Overview")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Industry", account_brief['INDUSTRY'])
                        with col2:
                            st.metric("Annual Revenue", f"€{account_brief['ANNUAL_REVENUE']:,}")
                        with col3:
                            st.metric("Total Activities", account_brief['TOTAL_ACTIVITIES'])
                        with col4:
                            st.metric("All Opportunities", account_brief['TOTAL_OPPORTUNITIES'])
                        
                        # Display opportunity-specific metrics
                        st.markdown("### 🎯 Opportunity Details")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Stage", opp_brief['STAGE_NAME'])
                        with col2:
                            st.metric("Deal Value", f"€{opp_brief['AMOUNT']:,.0f}")
                        with col3:
                            st.metric("Probability", f"{opp_brief['PROBABILITY']}%")
                        with col4:
                            st.metric("Close Date", opp_brief['CLOSE_DATE'])
                        
                        # Display AI-generated opportunity brief
                        st.markdown("### 📋 AI-Generated Opportunity Brief")
                        st.markdown(opp_brief['OPPORTUNITY_BRIEF'])
                        
                        # Get recent contacts using cached function (unchanged)
                        contacts_df = get_account_contacts(selected_account)
                        
                        if not contacts_df.empty:
                            # Format email addresses as mailto links
                            contacts_df['EMAIL'] = contacts_df['EMAIL'].apply(lambda x: f"mailto:{x}" if pd.notna(x) else x)
                            
                            st.markdown("### 👥 Key Contacts")
                            st.dataframe(
                                contacts_df, 
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "EMAIL": st.column_config.LinkColumn(
                                        "Email",
                                        help="Click to send email"
                                    )
                                }
                            )
                    else:
                        st.warning("Could not generate meeting brief. Please ensure the opportunity has associated activity data.")
                        
                except Exception as e:
                    st.error(f"Error generating meeting brief: {str(e)}")
    else:
        st.warning(f"No active opportunities found for {selected_account}. Please select an account with open opportunities.")


# ==================
# MEDDIC ANALYSIS TAB
# ==================
with tab4:
    st.info("AI-powered MEDDIC assessment across your opportunity pipeline.")
    
    if st.button("📊 Analyze MEDDIC Health"):
        with st.spinner("Analyzing opportunities for MEDDIC elements..."):
            try:
                meddic_query = f"""
                    SELECT
                        o.name AS opportunity_name,
                        a.name AS account_name,
                        o.stage_name,
                        o.amount,
                        o.close_date,
                        AI_FILTER(
                            PROMPT(
                                'Does this opportunity show clear metrics or ROI discussions? Look for financial impact, cost savings, revenue growth targets. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_metrics,
                        AI_FILTER(
                            PROMPT(
                                'Is there evidence of economic buyer engagement? Look for C-level involvement, budget approval discussions, or executive sponsorship. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_economic_buyer,
                        AI_FILTER(
                            PROMPT(
                                'Are decision criteria clearly defined? Look for evaluation requirements, technical specifications, or vendor comparison criteria. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_decision_criteria,
                        AI_FILTER(
                            PROMPT(
                                'Is the decision process mapped out? Look for timeline discussions, approval workflows, or procurement processes. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_decision_process,
                        AI_FILTER(
                            PROMPT(
                                'Are there clear pain points or business challenges identified? Look for problems, inefficiencies, or urgent business needs. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_implicated_pain,
                        AI_FILTER(
                            PROMPT(
                                'Is there evidence of a champion or internal advocate? Look for stakeholder support, internal selling, or advocacy behaviors. {{0}}',
                                (SELECT LISTAGG(description, ' | ') FROM {TASKS_TABLE} AS t WHERE t.what_id = o.id)
                            )
                        ) AS has_champion
                    FROM {OPPORTUNITIES_TABLE} AS o
                    JOIN {ACCOUNTS_TABLE} AS a ON o.account_id = a.id
                    WHERE NOT o.stage_name IN ('Closed Won', 'Closed Lost')
                    ORDER BY o.amount DESC
                    LIMIT 15
                """
                
                meddic_df = session.sql(meddic_query).to_pandas()
                
                if not meddic_df.empty:
                    # Calculate MEDDIC score
                    meddic_columns = ['HAS_METRICS', 'HAS_ECONOMIC_BUYER', 'HAS_DECISION_CRITERIA', 
                                    'HAS_DECISION_PROCESS', 'HAS_IMPLICATED_PAIN', 'HAS_CHAMPION']
                    
                    meddic_df['MEDDIC_SCORE'] = meddic_df[meddic_columns].sum(axis=1)
                    meddic_df['HEALTH_STATUS'] = meddic_df['MEDDIC_SCORE'].apply(
                        lambda x: 'Healthy' if x >= 5 else 'At Risk' if x >= 3 else 'Critical'
                    )
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        healthy_deals = len(meddic_df[meddic_df['HEALTH_STATUS'] == 'Healthy'])
                        st.metric("Healthy Deals", healthy_deals)
                    
                    with col2:
                        at_risk_deals = len(meddic_df[meddic_df['HEALTH_STATUS'] == 'At Risk'])
                        st.metric("At Risk Deals", at_risk_deals)
                    
                    with col3:
                        critical_deals = len(meddic_df[meddic_df['HEALTH_STATUS'] == 'Critical'])
                        st.metric("Critical Deals", critical_deals)
                    
                    # MEDDIC health distribution
                    health_summary = meddic_df['HEALTH_STATUS'].value_counts().reset_index()
                    health_summary.columns = ['Status', 'Count']
                    
                    fig_health = px.bar(
                        health_summary,
                        x='Status',
                        y='Count',
                        title='Deal Health Distribution',
                        color='Status',
                        color_discrete_map={'Healthy': 'green', 'At Risk': 'orange', 'Critical': 'red'}
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                    
                    # Detailed MEDDIC analysis
                    st.markdown("### 📋 Detailed MEDDIC Analysis")
                    
                    # Display deals with expandable details
                    for _, deal in meddic_df.iterrows():
                        score_color = "🟢" if deal['MEDDIC_SCORE'] >= 5 else "🟡" if deal['MEDDIC_SCORE'] >= 3 else "🔴"
                        
                        with st.expander(f"{score_color} {deal['OPPORTUNITY_NAME']} - Score: {deal['MEDDIC_SCORE']}/6"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Account:** {deal['ACCOUNT_NAME']}")
                                st.markdown(f"**Stage:** {deal['STAGE_NAME']}")
                                st.markdown(f"**Amount:** €{deal['AMOUNT']:,.0f}")
                                st.markdown(f"**Close Date:** {deal['CLOSE_DATE']}")
                            
                            with col2:
                                st.markdown("**MEDDIC Elements:**")
                                st.markdown(f"{'✅' if deal['HAS_METRICS'] else '❌'} Metrics: {'Yes' if deal['HAS_METRICS'] else 'No'}")
                                st.markdown(f"{'✅' if deal['HAS_ECONOMIC_BUYER'] else '❌'} Economic Buyer: {'Yes' if deal['HAS_ECONOMIC_BUYER'] else 'No'}")
                                st.markdown(f"{'✅' if deal['HAS_DECISION_CRITERIA'] else '❌'} Decision Criteria: {'Yes' if deal['HAS_DECISION_CRITERIA'] else 'No'}")
                                st.markdown(f"{'✅' if deal['HAS_DECISION_PROCESS'] else '❌'} Decision Process: {'Yes' if deal['HAS_DECISION_PROCESS'] else 'No'}")
                                st.markdown(f"{'✅' if deal['HAS_IMPLICATED_PAIN'] else '❌'} Implicated Pain: {'Yes' if deal['HAS_IMPLICATED_PAIN'] else 'No'}")
                                st.markdown(f"{'✅' if deal['HAS_CHAMPION'] else '❌'} Champion: {'Yes' if deal['HAS_CHAMPION'] else 'No'}")
                else:
                    st.info("No open opportunities found for MEDDIC analysis.")
                    
            except Exception as e:
                st.error(f"Error analyzing MEDDIC health: {str(e)}")

# ====================
# OUTREACH GENERATOR TAB
# ====================
with tab5:  
    st.info("Generate personalized follow-up emails based on recent interactions and account context.")
    
    # Get accounts with recent activity
    recent_accounts_query = f"""
        SELECT DISTINCT 
            a.name,
            MAX(t.activity_date) as last_activity
        FROM {ACCOUNTS_TABLE} a
        JOIN {TASKS_TABLE} t ON a.id = t.account_id
        GROUP BY a.name
        ORDER BY last_activity DESC
        LIMIT 20
    """
    
    recent_accounts = session.sql(recent_accounts_query).to_pandas()
    
    # Helper function to get contacts for selected account
    def get_contacts_for_account(account_name):
        """Get contacts for a specific account"""
        contacts_query = f"""
            SELECT 
                c.first_name || ' ' || c.last_name as contact_name,
                c.title,
                c.email,
                c.is_primary_contact
            FROM {CONTACTS_TABLE} c
            JOIN {ACCOUNTS_TABLE} a ON c.account_id = a.id
            WHERE a.name = '{account_name.replace("'", "''")}'
            ORDER BY c.is_primary_contact DESC, c.last_name
        """
        return session.sql(contacts_query).to_pandas()
    
    # Account selection outside form to allow dynamic updates
    selected_account_outreach = st.selectbox(
        "Select Account for Outreach:",
        options=recent_accounts['NAME'].tolist(),
        key="account_selector"
    )
    
    # Get contacts for selected account (this will update when account changes)
    contacts_df = get_contacts_for_account(selected_account_outreach)
    
    # Form for outreach generation
    with st.form("outreach_form"):
        # Contact selection and other form elements
        col1, col2 = st.columns(2)
        
        with col1:
            if not contacts_df.empty:
                # Create display options with name and title
                contact_options = []
                for _, contact in contacts_df.iterrows():
                    display_name = f"{contact['CONTACT_NAME']}"
                    if contact['TITLE']:
                        display_name += f" ({contact['TITLE']})"
                    if contact['IS_PRIMARY_CONTACT']:
                        display_name += " [Primary]"
                    contact_options.append(display_name)
                
                selected_contact_display = st.selectbox(
                    "Select Contact:",
                    options=contact_options
                )
                
                # Extract actual contact name for use in query
                selected_contact = contacts_df.iloc[contact_options.index(selected_contact_display)]['CONTACT_NAME']
            else:
                selected_contact_display = st.selectbox("Select Contact:", options=["No contacts found"], disabled=True)
                selected_contact = "Key Contact"
        
        with col2:
            # Move tone selection here to balance the layout
            tone = st.selectbox(
                "Tone:",
                ["Professional", "Friendly", "Executive", "Technical"]
            )
        
        # Outreach Type selection
        outreach_type = st.selectbox(
            "Outreach Type:",
            ["Follow-up after meeting", "Next steps confirmation", "Proposal follow-up", "Check-in email", "Meeting request"]
        )
        
        # Purpose and context guidance
        st.markdown("### 🎯 Outreach Purpose & Context")
        custom_context = st.text_area(
            "What's the main aim of this outreach? (Be specific about objectives)",
            placeholder="Example: 'I'd like to reach out to Magnus with the aim of understanding his technical challenges in integration and explore how our API solutions could address their current workflow bottlenecks.'",
            height=100
        )
        
        generate_outreach = st.form_submit_button("📧 Generate Outreach Email")
    
    if generate_outreach:
        with st.spinner("Generating personalized outreach email..."):
            try:
                # Escape single quotes to prevent SQL injection
                safe_account = selected_account_outreach.replace("'", "''")
                safe_contact = selected_contact.replace("'", "''")
                safe_outreach_type = outreach_type.replace("'", "''")
                safe_tone = tone.replace("'", "''")
                safe_custom_context = custom_context.replace("'", "''")
                
                # Get contact details for the selected contact (if not generic)
                contact_info = pd.DataFrame()
                contact_title = ""
                
                if safe_contact != "Key Contact":
                    contact_details_query = f"""
                        SELECT 
                            c.first_name || ' ' || c.last_name as contact_name,
                            c.title,
                            c.email
                        FROM {CONTACTS_TABLE} c
                        JOIN {ACCOUNTS_TABLE} a ON c.account_id = a.id
                        WHERE a.name = '{safe_account}'
                        AND c.first_name || ' ' || c.last_name = '{safe_contact}'
                        LIMIT 1
                    """
                    contact_info = session.sql(contact_details_query).to_pandas()
                
                # Build the AI prompt based on available contact info
                if not contact_info.empty and pd.notna(contact_info.iloc[0]['TITLE']):
                    contact_title = contact_info.iloc[0]['TITLE']
                
                safe_contact_title = contact_title.replace("'", "''")
                
                # Build the contact reference
                contact_reference = safe_contact
                if safe_contact_title:
                    contact_reference += f" ({safe_contact_title})"
                
                # Build context addition
                context_addition = ""
                if safe_custom_context:
                    context_addition = f" Focus on this specific objective: {safe_custom_context}"
                
                # Get account context and recent activities
                outreach_context_query = f"""
                    WITH recent_activities AS (
                        SELECT 
                            t.description,
                            t.subject,
                            t.activity_date,
                            t.type
                        FROM {TASKS_TABLE} t
                        JOIN {ACCOUNTS_TABLE} a ON t.account_id = a.id
                        WHERE a.name = '{safe_account}'
                        ORDER BY t.activity_date DESC
                        LIMIT 5
                    ),
                    account_context AS (
                        SELECT 
                            a.name as account_name,
                            a.industry,
                            a.annual_revenue
                        FROM {ACCOUNTS_TABLE} a
                        WHERE a.name = '{safe_account}'
                        LIMIT 1
                    )
                    SELECT 
                        ac.account_name,
                        ac.industry,
                        ac.annual_revenue,
                        SNOWFLAKE.CORTEX.COMPLETE(
                            'claude-4-sonnet',
                            CONCAT(
                                'Write a {safe_outreach_type} email in a {safe_tone} tone to {contact_reference} at ', 
                                ac.account_name, ' (', ac.industry, ' industry). ',
                                'Base the email on these recent interactions: ',
                                (SELECT LISTAGG(CONCAT(type, ': ', subject, ' - ', description), '; ') FROM recent_activities),
                                '. Make it personalized, relevant, and include appropriate next steps. Keep it concise (under 200 words). ',
                                'Include a clear subject line. Sign as the Snowball Analytics sales representative.{context_addition}'
                            )
                        ) as generated_email
                    FROM account_context ac
                """
                
                outreach_data = session.sql(outreach_context_query).to_pandas()
                
                if not outreach_data.empty:
                    context = outreach_data.iloc[0]
                    
                    st.success(f"Outreach Generated for {context['ACCOUNT_NAME']}")
                    
                    # Display contact info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**To:** {selected_contact}")
                        if not contact_info.empty:
                            st.markdown(f"**Title:** {contact_info.iloc[0]['TITLE'] if pd.notna(contact_info.iloc[0]['TITLE']) else 'N/A'}")
                        else:
                            st.markdown(f"**Title:** N/A")
                    with col2:
                        if not contact_info.empty:
                            st.markdown(f"**Email:** {contact_info.iloc[0]['EMAIL'] if pd.notna(contact_info.iloc[0]['EMAIL']) else 'N/A'}")
                        else:
                            st.markdown(f"**Email:** N/A")
                        st.markdown(f"**Company:** {context['ACCOUNT_NAME']}")
                    
                    # Display generated email
                    st.markdown("### 📧 Generated Email")
                    
                    email_content = context['GENERATED_EMAIL']
                    
                    # Make email editable
                    edited_email = st.text_area(
                        "Edit email before sending:",
                        value=email_content,
                        height=300,
                        key="email_editor"
                    )
                    
                    # Copy to clipboard button
                    if st.button("📋 Copy to Clipboard"):
                        st.success("Email copied to clipboard! (Note: In a real app, this would use clipboard API)")
                        
                else:
                    st.warning("Could not generate outreach for this account. Please ensure the account has recent activity data and try again.")
                    
            except Exception as e:
                st.error(f"Error generating outreach: {str(e)}") 