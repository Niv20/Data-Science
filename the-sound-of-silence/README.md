# The Sound of Silence

> *"People talking without speaking, people hearing without listening..."*

An exploration of the **"culture of silence"** in the ISSP (International Social Survey Programme) *Family and Changing Gender Roles* surveys. Instead of analyzing what people answered, this project focuses on **who refuses to answer** — respondents who choose "Can't choose," "Refuse," or leave sensitive questions blank.

## Overview

We analyze three ISSP survey waves on family and gender roles — **2002**, **2012**, and **2022** — to understand patterns of non-response across countries, topics, genders, and age groups, and to test whether silence can be predicted from demographic and behavioral data.

Silence is defined as choosing "Can't choose," "Refuse to answer," or a similar special missing-value code (e.g. `-8`, `-9` in ISSP questionnaires) rather than providing a substantive answer.

## What's inside

The analysis (`The Sound of Silence.ipynb`) walks through:

1. **Culture of Silence by Country** — comparing silence rates across countries, from high-privacy Western nations (Australia, Germany, Netherlands, US, Japan) to low-silence countries (Slovakia, South Africa, Philippines).
2. **Silence Rates by Question Category** — which topics people find hardest to answer (work-family balance and household chores top the list; wellbeing and parental roles are the most comfortable).
3. **Country-Level Patterns** — including a Western vs. post-communist European bloc comparison, Z-score normalization to find each country's most *unusually* avoided topic, and deep dives into specific anomalies:
   - **The Netherlands**, where general opinions are answered openly but personal work-family conflict questions see silence spike to 40–47%.
   - **Japan**, where "Women Leadership" questions show the highest normalized silence (Z-score of 4.2) of any country/topic pair, broken down by gender and age.
   - **Iceland**, an outlier among otherwise-open Nordic countries when it comes to wellbeing questions.
4. **Detailed Gender Attitudes Breakdown** — a 2002–2012–2022 comparison of five gender-attitude questions in Israel, highlighting a dramatic spike in silence in 2012, plausibly linked to the 2011 social protests and the run-up to that year's Knesset elections.
5. **Predictive Modeling** — several attempts to predict who stays silent:
   - **Model A** (demographics: age, education, social status) — R² ≈ 2%.
   - **Model B** (household structure) — R² < 1%.
   - **Model C** (cross-category silence, logistic regression) — the best performer, ~79.5% accuracy, showing that silence on one topic strongly predicts silence on related topics.
   - **KNN classifier** on demographic features — ~60% accuracy, modestly above the 55% baseline.

## Key takeaway

Demographics alone barely predict who stays silent. The strongest signal is **behavioral**: people who stay silent on one sensitive topic tend to stay silent on related topics too. Silence looks less like a socioeconomic trait and more like a cultural and psychological phenomenon — shaped by privacy norms, survey fatigue, and topic sensitivity rather than by who someone is on paper.

## Repository contents

| File | Description |
|---|---|
| `The Sound of Silence.ipynb` | Main analysis notebook |
| `The Sound of Silence.pdf` | Exported PDF version of the notebook |
| `Silence Culture Analysis.pptx` | Presentation summarizing the findings |
| `2002Question.pdf`, `2012Question.pdf`, `2022Question.pdf` | ISSP questionnaires for each survey wave |

## Data source

[ISSP — International Social Survey Programme](https://issp.org/), *Family and Changing Gender Roles* modules (2002, 2012, 2022).

## Links

- Project page: [nivniv.dev/projects/programming/the-sound-of-silence](https://www.nivniv.dev/projects/programming/the-sound-of-silence)
- ["The Sound of Silence" (Simon & Garfunkel)](https://youtu.be/NAEppFUWLfc)
