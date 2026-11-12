#!/bin/bash
# Results of the additional analyses (no training): summaries and combined tables.

# Load the shared settings and helper functions (summarize, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

summarize CBP  "pain"            # combined_pain_CBP_tables.csv
summarize NP   "pain"            # combined_pain_NP_tables.csv
summarize LOSO "age pain"        # combined_{age,pain}_LOSO_tables.csv
summarize RAW  "ageRAW painRAW"  # combined_{ageRAW,painRAW}_tables.csv
