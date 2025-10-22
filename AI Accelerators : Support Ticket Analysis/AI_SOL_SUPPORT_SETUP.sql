--============================================
-- Getting started w/ Support Ticket Analysis
--============================================
-- This SQL Script will help to set up the Support Ticket Analysis demo
-- It creates the necessary schema, loads sample data from GitHub, and sets up the notebook
-- 
-- Data Source: https://github.com/sfc-gh-tchristian/AI-Accelerators/blob/main/AI%20Accelerators%20%3A%20Support%20Ticket%20Analysis/support_data.csv
-- NOTE: This demo uses synthetic support ticket data with realistic patterns

-- ============================================
-- STEP 1: Create schema and prepare for data
-- ============================================
CREATE SCHEMA IF NOT EXISTS AI_SOL.SUPPORT;
USE AI_SOL.SUPPORT;

-- Create a stage for files
CREATE STAGE IF NOT EXISTS AI_SOL.SUPPORT.support_files_stage
    DIRECTORY = ( ENABLE = true )
    ENCRYPTION = ( TYPE = 'SNOWFLAKE_SSE' );

-- ============================================
-- STEP 2: Load demo data from GitHub
-- ============================================
-- Copy the sample support ticket data from the GitHub repository
COPY FILES
  INTO @support_files_stage
  FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : Support Ticket Analysis/'
  PATTERN='.*support_data[.]csv';
  
LS @support_files_stage;

-- ============================================
-- STEP 3: Create raw table and load data
-- ============================================
-- Create raw table for support ticket data
CREATE OR REPLACE TABLE AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS (
    TICKET_ID VARCHAR,
    CUSTOMER_ID VARCHAR,
    SUBMIT_DATE TIMESTAMP_NTZ,
    TICKET_DESCRIPTION TEXT,
    PRIORITY VARCHAR,
    STATUS VARCHAR,
    CUSTOMER_TIER VARCHAR,
    CHANNEL VARCHAR,
    PRODUCT_AREA VARCHAR,
    SENTIMENT VARCHAR,
    CLASSIFICATION VARCHAR,
    RESPONSE_TIME NUMBER,
    RESOLUTION_TIME NUMBER
);

-- Load data from staged CSV file
COPY INTO AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS
FROM @support_files_stage
FILES = ('support_data.csv')
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

-- ============================================
-- STEP 4: Verify data loaded successfully
-- ============================================
SELECT 'Support Tickets' as table_name, COUNT(*) as row_count 
FROM AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS;

-- Sample the data
SELECT * FROM AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS LIMIT 5;

-- Check date range
SELECT 
    MIN(SUBMIT_DATE) as earliest_ticket,
    MAX(SUBMIT_DATE) as latest_ticket,
    COUNT(DISTINCT CUSTOMER_ID) as unique_customers
FROM AI_SOL.SUPPORT.RAW_SUPPORT_TICKETS;

-- ============================================
-- STEP 5: Create the notebook
-- ============================================
-- If you have the notebook in a Git repository, you can create it directly:

CREATE OR REPLACE NOTEBOOK AI_ACCELERATORS_SUPPORT_TICKET_ANALYSIS
     FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators : Support Ticket Analysis/' 
         MAIN_FILE = 'AI Accelerators _ Support Ticket Analysis.ipynb' 
         COMPUTE_POOL = SYSTEM_COMPUTE_POOL_CPU --replace with CPU pool (if needed)
         RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
         ;
        
ALTER NOTEBOOK AI_ACCELERATORS_SUPPORT_TICKET_ANALYSIS ADD LIVE VERSION FROM LAST;

SELECT 'Notebook created successfully' as message;

-- ============================================
-- Setup Complete! 
-- ============================================
-- You can now:
-- 1. Open the AI Accelerators Support Ticket Analysis notebook
-- 2. Use Cortex Analyst to query the semantic view with natural language
-- 3. Build Streamlit apps using the support ticket data and metadata
-- 4. Create agents that leverage the search service and AI functions
--
-- Example natural language queries for Cortex Analyst:
-- - "What are the most common issues causing system outages?"
-- - "Show me escalation rates by customer tier"
-- - "Which product areas have the most negative sentiment?"
-- - "What's the average resolution time for critical tickets?"

