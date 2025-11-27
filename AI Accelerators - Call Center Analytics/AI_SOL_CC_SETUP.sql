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

CREATE TABLE IF NOT EXISTS AI_SOL.CALL_CENTERS.TRANSCRIBED_AUDIO (
    AUDIO_FILE VARIANT,
    RAW_TRANSCRIPT VARIANT
);

COPY INTO AI_SOL.CALL_CENTERS.TRANSCRIBED_AUDIO
FROM @audio_files_stage
FILE_FORMAT = (
    TYPE = 'CSV'
    FIELD_DELIMITER = '\t'
    SKIP_HEADER = 0
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    ESCAPE = NONE
    ESCAPE_UNENCLOSED_FIELD = NONE
    ENCODING = 'UTF8'
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
    TRIM_SPACE = FALSE
);

-- ============================================
-- STEP 3: Create the notebook
-- ============================================


CREATE OR REPLACE NOTEBOOK AI_ACCELERATORS_CALL_CENTER_ANALYTICS
    FROM '@AI_SOL.PUBLIC.ACCELERATOR_REPO/branches/main/AI Accelerators - Call Center Analytics/' 
        MAIN_FILE = 'AI Accelerators _ Call Center Analytics.ipynb' 
        COMPUTE_POOL = SYSTEM_COMPUTE_POOL_CPU --replace with CPU pool (if needed)
        RUNTIME_NAME = 'SYSTEM$BASIC_RUNTIME'
        ;
        
ALTER NOTEBOOK AI_ACCELERATORS_CALL_CENTER_ANALYTICS ADD LIVE VERSION FROM LAST;