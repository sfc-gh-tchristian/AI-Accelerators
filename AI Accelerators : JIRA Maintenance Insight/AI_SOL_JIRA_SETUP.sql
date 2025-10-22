--============================================
-- Getting started w/ Manufacturing Analytics (JIRA)
--============================================
-- This SQL Script will help to set up the JIRA Manufacturing Analytics demo
-- It creates the necessary schema, loads sample data from GitHub, and sets up the notebook
-- 
-- Data Source: https://github.com/sfc-gh-tchristian/AI-Accelerators/tree/main/AI%20Accelerators%20%3A%20JIRA%20Maintenance%20Insight/data
-- NOTE: This demo uses synthetic manufacturing data with realistic failure patterns

-- ============================================
-- STEP 1: Create schema and prepare for data
-- ============================================
CREATE SCHEMA IF NOT EXISTS AI_SOL.JIRA;
USE AI_SOL.JIRA;

-- Create a stage for any external files if needed
CREATE STAGE IF NOT EXISTS AI_SOL.JIRA.jira_files_stage
    DIRECTORY = ( ENABLE = true )
    ENCRYPTION = ( TYPE = 'SNOWFLAKE_SSE' );

-- ============================================
-- STEP 2: Load demo data from GitHub
-- ============================================
-- Copy the sample JIRA manufacturing data from the GitHub repository
COPY FILES
  INTO @jira_files_stage
  FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : JIRA Maintenance Insight/data/'
  PATTERN='.*[.]csv';
  
LS @jira_files_stage;

-- ============================================
-- STEP 3: Create raw tables and load data
-- ============================================
-- Create raw tables for JIRA data
CREATE OR REPLACE TABLE AI_SOL.JIRA.RAW_JIRA_ISSUES (
    KEY VARCHAR,
    ID VARCHAR,
    PROJECT_CODE VARCHAR,
    PROJECT_NAME VARCHAR,
    ASSIGNEE VARCHAR,
    CREATOR VARCHAR,
    CREATED TIMESTAMP_NTZ,
    CHANNEL VARCHAR,
    OWNER VARCHAR,
    REPORTER VARCHAR,
    LABELS TEXT,
    SUMMARY VARCHAR,
    STATUS VARCHAR,
    STATUS_CATEGORY VARCHAR,
    PRIORITY VARCHAR,
    RESOLVED_TS TIMESTAMP_NTZ
);

CREATE OR REPLACE TABLE AI_SOL.JIRA.RAW_JIRA_PROJECTS (
    PROJECT_CODE VARCHAR,
    PROJECT_NAME VARCHAR
);

CREATE OR REPLACE TABLE AI_SOL.JIRA.RAW_JIRA_USERS (
    ID VARCHAR,
    DISPLAY_NAME VARCHAR,
    ROLE VARCHAR,
    TEAM VARCHAR,
    LOCATION VARCHAR,
    EMAIL VARCHAR
);

-- Load data from staged CSV files
COPY INTO AI_SOL.JIRA.RAW_JIRA_ISSUES
FROM @jira_files_stage
FILES = ('issues.csv')
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

COPY INTO AI_SOL.JIRA.RAW_JIRA_PROJECTS
FROM @jira_files_stage
FILES = ('projects.csv')
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

COPY INTO AI_SOL.JIRA.RAW_JIRA_USERS
FROM @jira_files_stage
FILES = ('users.csv')
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

-- ============================================
-- STEP 4: Create cleaned/production tables
-- ============================================
-- Create production-ready views or tables from the raw data

CREATE OR REPLACE TABLE AI_SOL.JIRA.ISSUES AS
SELECT 
    KEY,
    ID,
    PROJECT_CODE,
    PROJECT_NAME,
    ASSIGNEE,
    CREATOR,
    CREATED,
    CHANNEL,
    OWNER,
    REPORTER,
    LABELS,
    SUMMARY,
    STATUS,
    STATUS_CATEGORY,
    PRIORITY,
    RESOLVED_TS
FROM AI_SOL.JIRA.RAW_JIRA_ISSUES;

CREATE OR REPLACE TABLE AI_SOL.JIRA.PROJECTS AS
SELECT 
    PROJECT_CODE,
    PROJECT_NAME
FROM AI_SOL.JIRA.RAW_JIRA_PROJECTS;

CREATE OR REPLACE TABLE AI_SOL.JIRA.USERS AS
SELECT 
    ID,
    DISPLAY_NAME,
    ROLE,
    TEAM,
    LOCATION,
    EMAIL
FROM AI_SOL.JIRA.RAW_JIRA_USERS;

-- ============================================
-- STEP 5: Verify data loaded successfully
-- ============================================
SELECT 'Issues' as table_name, COUNT(*) as row_count FROM AI_SOL.JIRA.ISSUES
UNION ALL
SELECT 'Projects', COUNT(*) FROM AI_SOL.JIRA.PROJECTS
UNION ALL
SELECT 'Users', COUNT(*) FROM AI_SOL.JIRA.USERS;

-- Sample the data
SELECT * FROM AI_SOL.JIRA.ISSUES LIMIT 5;
SELECT * FROM AI_SOL.JIRA.PROJECTS;
SELECT * FROM AI_SOL.JIRA.USERS LIMIT 5;

-- ============================================
-- STEP 6: Create the notebook
-- ============================================
-- If you have the notebook in a Git repository, you can create it directly:

CREATE OR REPLACE NOTEBOOK AI_ACCELERATORS_MANUFACTURING_ANALYTICS
    FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : JIRA Maintenance Insight/' 
        MAIN_FILE = 'AI Accelerators _ Manufacturing Analytics.ipynb' 
        COMPUTE_POOL = SYSTEM_COMPUTE_POOL_CPU --replace with CPU pool (if needed)
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
        ;
        
 ALTER NOTEBOOK AI_ACCELERATORS_MANUFACTURING_ANALYTICS ADD LIVE VERSION FROM LAST;

SELECT 'Notebook created successfully' as message;

-- ============================================
-- Setup Complete! 
-- ============================================
-- You can now:
-- 1. Open the AI Accelerators Manufacturing Analytics notebook
-- 2. Use Cortex Analyst to query the semantic view with natural language
-- 3. Build Streamlit apps using the JIRA data and metadata
-- 4. Create agents that leverage the search service and AI functions
--
-- Example natural language queries for Cortex Analyst:
-- - "What equipment has the most downtime?"
-- - "Show me safety issues by department"
-- - "Which failure types are most common in Plant A?"
-- - "What's the average resolution time by urgency level?"

