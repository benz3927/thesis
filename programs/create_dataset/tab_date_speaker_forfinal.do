clear
import delimited "fomc_transcript/data/processed/sets/processed_final.csv", clear

log using "fomc_transcript/misc/log_search_names_dup.smcl", replace


bigtab unique_id date

log close
