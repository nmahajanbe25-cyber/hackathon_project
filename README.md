# Candidate Ranking System

## Overview

This project ranks candidates for a Machine Learning Engineering role based on their experience, skills, job titles, and profile information.

## Files

* `rank.py` – Main ranking algorithm
* `submission.csv` – Ranked candidate output
* `.gitignore` – Excludes large dataset files from version control

## Approach

The ranking system evaluates candidates using:

* Machine Learning experience
* Relevant skills and technologies
* Job title relevance
* Career progression and seniority
* Search, retrieval, recommendation, and ranking experience

Candidates are scored and sorted to produce the final ranking.

## How to Run

```bash
python rank.py
```

The script generates a ranked list of candidates in `submission.csv`.

## Notes

The input dataset provided by the organizers (`candidates.jsonl`) is not included in this repository because it exceeds GitHub's file size limit.

```
```
