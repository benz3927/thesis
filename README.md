# Do FOMC Transcripts Reveal More Dissent Than Votes?

**Using NLP to Measure Regional Influence on Monetary Policy Deliberation**

Benjamin Zhao | Professor Ann Owen, Advisor | Hamilton College | May 2026

---

## Overview

This paper tests whether regional economic conditions show up more strongly in what Federal Reserve district presidents *say* than in how they *vote*. I use GPT-4o to score each district president's remarks in FOMC transcripts on a −10 (hawkish) to +10 (dovish) scale, then run Bobrov-style regressions relating these scores to district-level unemployment gaps.

**Key findings:**
- Speech scores respond significantly to regional unemployment gaps (β = +0.494, p = 0.036), while the vote effect is only marginal (β = +0.018, p = 0.051)
- Interaction coefficients are consistently positive across all four scoring versions, suggesting the effect strengthened after Greenspan's departure (2006)
- Results are robust across 4 scoring versions (GPT v3, v7, v8, Claude Sonnet)

## Validation

| Metric | GPT v8 | Keyword Baseline |
|--------|--------|-----------------|
| ROC-AUC | 0.741 | 0.580 |
| Direction Accuracy | 83.8% | 34.3% |
| Cross-model (Claude) | r = 0.838 | — |

## Data

- **FOMC Transcripts:** 216 meetings, 1994–2020 (parsed from Fed website PDFs)
- **Dissent Votes:** Thornton & Wheelock (2014), extended through 2020
- **Unemployment:** County-level BLS Local Area Unemployment Statistics (LAUS) data, aggregated to Fed districts

Data files are not included in this repository due to size. Raw transcripts are available from the [Federal Reserve](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm).

## Scripts

### Scoring
| Script | Description |
|--------|-------------|
| `score_v8.py` | GPT-4o scoring (final version) — scores each speaker-meeting on −10 to +10 scale |
| `claude_score_dissent.py` | Claude Sonnet scoring (robustness check, same prompt) |

### Analysis
| Script | Description |
|--------|-------------|
| `reg.py` | Bobrov-style regressions: votes vs speech, with speaker + meeting FE |
| `validate.py` | ROC-AUC, direction accuracy, within-person validation across all versions |
| `eda.py` | Descriptive statistics, figures, and appendix tables |

### Data Preparation
| Script | Description |
|--------|-------------|
| `get_transcripts.py` | Download and parse FOMC transcript PDFs |
| `get_unemployment.py` | Build district-level unemployment from county BLS LAUS data |
| `examples.py` | Extract scored transcript excerpts for Appendix D |

### Transcript Processing (Professor Owen)
The `programs/`, `new-programs/`, `fomc_transcript/`, and `htmls/` directories contain transcript extraction and processing scripts from Professor Owen's earlier work on this dataset. These handle the initial HTML-to-CSV parsing of FOMC transcripts, attendance extraction, and variable construction that produced the base dataset used in this analysis.

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: `openai`, `pandas`, `numpy`, `statsmodels`, `matplotlib`, `scikit-learn`, `openpyxl`

## Citation

```
Zhao, Benjamin (2026). "Do FOMC Transcripts Reveal More Dissent Than Votes? 
Using NLP to Measure Regional Influence on Monetary Policy Deliberation." 
Senior Thesis, Hamilton College.
```

## References

- Bobrov, Kamdar, and Ulate (2025). "Regional Dissent." *AER: Insights*, 7(2), 268–284.
- Tsang and Yang (2025). "Agree to Disagree." *JEDC*, 180, 105197.
- Hansen and Kazinnik (2024). "Can ChatGPT Decipher Fedspeak?"