--============================================
-- Getting started w/ Call Center Analytics
--============================================
-- This SQL Script will help to load the example post-transcript data directly from Git
-- And set up the notebook for the quickstart
-- NOTE: if you have trouble creating a GIT integration, feel free to download the CSV and load to a table using the UI wizard.

-- ============================================
-- STEP 1: Create stage for files (backup text or audio if available)
-- ============================================
CREATE SCHEMA IF NOT EXISTS AI_SOL.CALL_CENTERS;
USE AI_SOL.CALL_CENTERS;

CREATE STAGE IF NOT EXISTS AI_SOL.CALL_CENTERS.audio_files_stage -- if you already have an external file location (eg. S3) with audios, point this there.
    DIRECTORY = ( ENABLE = true )
    ENCRYPTION = ( TYPE = 'SNOWFLAKE_SSE' );

-- ============================================
-- STEP 2: Add demo files from stage
-- NOTE: I'm currently mimicking the audio output step
-- ============================================
COPY FILES
  INTO @audio_files_stage
  FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators - Call Center Analytics/'
  PATTERN='.*[.]csv';
  
LS @AUDIO_FILES_STAGE;

-- ============================================
-- STEP 3: Create the notebook
-- ============================================

CREATE OR REPLACE NOTEBOOK AI_ACCELERATORS_CALL_CENTER_ANALYTICS
    FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators - Call Center Analytics/' 
        MAIN_FILE = 'AI Accelerators _ Call Center Analytics.ipynb' 
        QUERY_WAREHOUSE = TC_WH --replace with your warehouse :)
        COMPUTE_POOL = SYSTEM_COMPUTE_POOL_CPU --replace with CPU pool (if needed)
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
        ;
        
ALTER NOTEBOOK AI_ACCELERATORS_CALL_CENTER_ANALYTICS ADD LIVE VERSION FROM LAST;