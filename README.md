# AI-Accelerators
Accelerators to show the full suite of Cortex features across real life use cases.

To quickly access these Accelerators - simply enable Git integration with this repository following these steps:
```sql
USE ROLE ACCOUNTADMIN; -- or any role with API integration privileges
CREATE DATABASE IF NOT EXISTS AI_SOL;

CREATE OR REPLACE API INTEGRATION git_api_integration
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/') -- feel free to scope this
    ENABLED = TRUE
    COMMENT = 'API Integration for Git repository access';

CREATE OR REPLACE GIT REPOSITORY AI_SOL.PUBLIC.ACCELERATOR_REPO
    API_INTEGRATION = git_api_integration
    ORIGIN = 'https://github.com/sfc-gh-tchristian/AI-Accelerators'
    COMMENT = 'Git repository containing AI accelerators';
```

At this point - you have access to several notebooks, dummy data, sample Streamlits, and agents you can access!

Follow the instructions below to get started.

## Call Center Analytics
```sql
EXECUTE IMMEDIATE FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators - Call Center Analytics/AI_SOL_CC_SETUP.sql';
```

<img width="654" height="310" alt="image" src="https://github.com/user-attachments/assets/58b10d62-273a-420d-88bf-0d6695af4517" />

## Manufacturing and JIRA Assistant
```sql
EXECUTE IMMEDIATE FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : JIRA Maintenance Insight/AI_SOL_JIRA_SETUP.sql';
```

- 🔍 Root Cause Analysis: AI-driven pattern recognition to identify recurring problems and their sources
- ⚡ Downtime Prevention: Predictive insights to prevent equipment failures before they occur
- 🔧 Maintenance Optimization: Intelligent scheduling based on failure patterns and asset health
- 📊 Compliance Monitoring: Automated tracking of safety and quality requirements
- 🏆 Process Excellence: Identify best practices and replicate high-performing operations
- 📈 Operational Efficiency: Real-time insights into production line performance and bottlenecks


## Sales Insight

<img width="389" height="212" alt="image" src="https://github.com/user-attachments/assets/93ad817d-f28e-4719-9b13-6ab3fe500c5b" />

Using AISQL to transform traditional CRM data into actionable sales intelligence such as:
- 🎯 Account Prioritization: AI-driven scoring to identify high-value opportunities and at-risk accounts
- 🤝 Meeting Preparation: Instant context and relationship history for prospect meetings
- 🏆 Competitive Intelligence: Extract competitor mentions and sentiment from sales activities
- 📈 MEDDIC Analysis: Automated identification of budget, timeline, and decision-maker signals
- ✉️ Personalized Outreach: Generate contextual follow-ups based on recent interactions
- 📊 Pipeline Health: Predict deal outcomes and recommended next actions


## Customer Support Analysis

```sql
EXECUTE IMMEDIATE FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : JIRA Maintenance Insight/AI_SOL_SUPPORT_SETUP.sql';
```

<img width="389" height="212" alt="image" src="https://github.com/user-attachments/assets/073d8e88-7a86-4106-ae48-fc709219a251" />

Demonstrating how Cortex can be used for RCA, key emerging impacts, as well as related case insight.
This solution ideally uses data from sources such as Zendesk, Salesforce, Hubspot, and ServiceNow.

Assets include:
- A notebook for analysis and tool configuration (Search & Semantic View)
- A streamlit app demonstrating AI integrations alongside BI & reporting - as well as case finding.
