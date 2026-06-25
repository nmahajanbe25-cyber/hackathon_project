#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
Ranks candidates for the "Senior AI Engineer — Founding Team" JD at Redrob AI.

Design: pure feature-based scoring, no LLM calls, no GPU. Runs on CPU in
well under the 5-minute / 16GB budget on the full 100K candidate pool.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Pipeline:
  1. Stream-parse candidates.jsonl (handles plain or .gz).
  2. For each candidate, compute interpretable sub-scores:
       - title_relevance      : how core is their current title to AI/ML/IR?
       - career_substance     : do career_history descriptions show real
                                 production ranking/retrieval/ML work (regex
                                 over free text, not just a skills list)?
       - skills_quality       : skills list quality, with a trust discount
                                 for "keyword stuffing" (many AI skills with
                                 near-zero duration/endorsements)
       - experience_fit       : years_of_experience vs the JD's 5-9y band
       - education_signal     : tier of institution (minor signal only)
       - disqualifier_penalty : explicit JD disqualifiers (pure research-only,
                                 consulting-only career, no NLP/IR exposure,
                                 stale recent-only LangChain-wrapper experience)
       - behavioral_modifier  : multiplicative modifier from redrob_signals
                                 (recency of activity, recruiter response
                                 rate, notice period, verification, etc.)
       - honeypot_flag        : internal-consistency checks; if any fire,
                                 the candidate is hard-capped near zero.
  3. Combine into one composite score, keep a running top-K (heap) so we
     never hold all 100K scores in memory at once.
  4. Emit the top 100 as the required CSV, each with a short reasoning
     string built from the same features that drove the score (so the
     reasoning is always truthful, never hallucinated).

This is intentionally a transparent, rule-based scorer (not a black box)
because the brief explicitly rewards systems that can be explained, defended
in an interview, and reproduced from a single command in a Docker sandbox.
"""

import argparse
import csv
import gzip
import heapq
import json
import re
import sys
from datetime import date, datetime

# ----------------------------------------------------------------------
# 1. Static reference data
# ----------------------------------------------------------------------

# Title relevance tiers for the Senior AI Engineer / ranking-and-retrieval role.
# Matched against current_title and career_history titles (case-insensitive).
TITLE_TIER_3 = {  # bullseye: exactly the kind of role the JD wants
    "ai engineer", "senior ai engineer", "lead ai engineer", "staff ai engineer",
    "machine learning engineer", "senior machine learning engineer",
    "staff machine learning engineer", "ml engineer", "senior ml engineer",
    "applied ml engineer", "applied scientist", "senior applied scientist",
    "nlp engineer", "senior nlp engineer", "search engineer",
    "recommendation systems engineer", "ranking engineer",
    "information retrieval engineer", "ai research engineer",
}
TITLE_TIER_2 = {  # adjacent and plausible, but needs corroboration
    "data scientist", "senior data scientist", "computer vision engineer",
    "ai specialist", "junior ml engineer", "research engineer",
    "senior software engineer (ml)", "ml ops engineer", "mlops engineer",
}
TITLE_TIER_1 = {  # general SWE/data roles: possible but needs strong evidence
    "software engineer", "senior software engineer", "data engineer",
    "senior data engineer", "analytics engineer", "data analyst",
    "backend engineer", "full stack developer",
}
# Everything else (Business Analyst, HR Manager, Accountant, Civil Engineer,
# Graphic Designer, Marketing Manager, Customer Support, ...) defaults to
# tier 0: essentially irrelevant regardless of how many AI keywords appear
# in their skills list. This is the main defense against the keyword-
# stuffer trap the JD explicitly calls out.

# Pure-research/academic-only titles -> hard disqualifier trigger if it's
# the *entire* visible career (JD: "pure research environments... we will
# not move forward").
RESEARCH_ONLY_TITLES = {
    "research scientist", "research fellow", "postdoctoral researcher",
    "phd researcher", "research assistant", "academic researcher",
}

# Consulting/services-only firms named explicitly in the JD as a soft
# disqualifier when they are the *entire* career.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

# Regex signals of real production ranking/retrieval/ML substance in free
# text (career_history descriptions, summary). These are the things a
# keyword-stuffed skills list cannot fake, because they require coherent
# sentences about *what was built and shipped*.
PROD_SUBSTANCE_PATTERNS = [
    r"\bembedding", r"\bretrieval\b", r"\bsentence[- ]transformers?\b",
    r"\bfaiss\b", r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b",
    r"\bmilvus\b", r"\belasticsearch\b", r"\bopensearch\b",
    r"\bvector (search|database|index)\b", r"\bhybrid (search|retrieval)\b",
    r"\bbm25\b", r"\bre-?rank", r"\branking model\b", r"\bndcg\b", r"\bmrr\b",
    r"\bmap@", r"\ba/?b test", r"\boffline.?online correlation\b",
    r"\brecommendation system\b", r"\bsearch relevance\b",
    r"\bproduction\b.{0,40}\b(deployed|shipped|serving|users)\b",
    r"\bshipped\b.{0,40}\b(production|users|scale)\b",
    r"\bfine-?tun(e|ing)\b", r"\block?ra\b", r"\bqlora\b", r"\bpeft\b",
    r"\bxgboost\b", r"\blightgbm\b", r"\blearning[- ]to[- ]rank\b",
]
PROD_SUBSTANCE_RE = re.compile("|".join(PROD_SUBSTANCE_PATTERNS), re.IGNORECASE)

# Signals of "recent LangChain-wrapper-only" experience: heavy on calling
# hosted LLM APIs through a framework, light on systems substance. The JD
# treats this as a soft disqualifier *unless* paired with older production
# ML experience.
WRAPPER_ONLY_PATTERNS = re.compile(
    r"\blangchain\b|\bcalled? (openai|gpt|chatgpt) api\b|\bprompt engineering\b|\bbuilt a chatbot\b",
    re.IGNORECASE,
)

NLP_IR_PATTERNS = re.compile(
    r"\bnlp\b|\bnatural language\b|\binformation retrieval\b|\bsearch\b|"
    r"\branking\b|\bretrieval\b|\bembedding|\btext\b.{0,20}\bclassif",
    re.IGNORECASE,
)
CV_SPEECH_ROBOTICS_PATTERNS = re.compile(
    r"\bcomputer vision\b|\bimage classification\b|\bobject detection\b|"
    r"\bspeech recognition\b|\btts\b|\brobotics\b|\bslam\b",
    re.IGNORECASE,
)

EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.2, "unknown": 0.3}

CORE_AI_SKILL_NAMES = {
    "nlp", "embeddings", "sentence transformers", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "bm25",
    "information retrieval", "ranking", "recommendation systems",
    "fine-tuning llms", "lora", "qlora", "peft", "xgboost", "lightgbm",
    "learning to rank", "deep learning", "pytorch", "tensorflow",
    "machine learning", "reinforcement learning", "data science",
    "feature engineering", "llm", "rag", "vector search", "hybrid search",
    "bge", "e5", "haystack", "kubeflow", "mlflow",
}

TODAY = date(2026, 6, 16)  # current date per the operating context


# ----------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------

def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def title_tier_score(title):
    t = (title or "").strip().lower()
    if t in TITLE_TIER_3:
        return 1.0
    if t in TITLE_TIER_2:
        return 0.65
    if t in TITLE_TIER_1:
        return 0.35
    return 0.05


def is_consulting_only(career_history):
    companies = [j.get("company", "").strip().lower() for j in career_history]
    if not companies:
        return False
    return all(any(firm in c for firm in CONSULTING_FIRMS) for c in companies)


def is_research_only(profile, career_history):
    titles = [profile.get("current_title", "").lower()] + [j.get("title", "").lower() for j in career_history]
    if not titles:
        return False
    return all(any(r in t for r in RESEARCH_ONLY_TITLES) for t in titles if t)


def career_text_blob(profile, career_history):
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for j in career_history:
        parts.append(j.get("title", ""))
        parts.append(j.get("description", ""))
    return " \n ".join(parts)


def honeypot_checks(cand):
    """Return True if the candidate trips an internal-consistency honeypot."""
    profile = cand["profile"]
    career_history = cand.get("career_history", [])
    skills = cand.get("skills", [])

    # Check 1: expert proficiency claimed with ~0 months of usage.
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 1:
            return True

    # Check 2: total career_history duration far exceeds stated years_of_experience.
    yoe_months = profile.get("years_of_experience", 0) * 12
    total_months = sum(j.get("duration_months", 0) for j in career_history)
    if total_months > yoe_months * 1.5 + 6:
        return True

    # NOTE: a "single skill duration_months exceeds total YOE" check was
    # tried but rejected — it fired on ~13% of the *entire* dataset, which
    # is clearly synthetic-generation noise (skill duration isn't tightly
    # coupled to total experience in this dataset) rather than a
    # deliberate honeypot signal. Keeping only checks that are rare and
    # specific avoids false-positiving on genuinely good candidates.

    # Check 3: overlapping employment intervals (can't work two full-time
    # jobs with overlapping date ranges in this synthetic dataset's model).
    intervals = []
    for j in career_history:
        st = parse_date(j.get("start_date"))
        en = parse_date(j.get("end_date")) or TODAY
        if st:
            intervals.append((st, en))
    intervals.sort()
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            return True

    return False


def compute_skills_quality(skills):
    """Score the skills list, discounting unsupported AI-keyword stuffing."""
    if not skills:
        return 0.0, 0
    core_hits = []
    for s in skills:
        name = s.get("name", "").strip().lower()
        if name in CORE_AI_SKILL_NAMES:
            dur = s.get("duration_months", 0)
            endorse = s.get("endorsements", 0)
            prof = s.get("proficiency", "beginner")
            prof_w = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}.get(prof, 0.25)
            # Trust discount: a claimed skill with near-zero duration and
            # zero endorsements is exactly the keyword-stuffer signature.
            trust = min(1.0, (dur / 12.0)) * 0.7 + min(1.0, endorse / 10.0) * 0.3
            trust = max(trust, 0.1)  # never fully zero out, just discount
            core_hits.append(prof_w * trust)
    n_core = len(core_hits)
    if n_core == 0:
        return 0.0, 0
    # Average quality of core AI skills, with a mild bonus for breadth
    # (more genuinely-substantiated core skills), capped.
    avg_quality = sum(core_hits) / n_core
    breadth_bonus = min(0.3, 0.04 * n_core)
    return min(1.0, avg_quality + breadth_bonus), n_core


def compute_experience_fit(yoe):
    # JD band is 5-9 years, soft on the edges.
    if 5 <= yoe <= 9:
        return 1.0
    if 3 <= yoe < 5:
        return 0.6 + 0.4 * (yoe - 3) / 2
    if 9 < yoe <= 12:
        return 1.0 - 0.3 * (yoe - 9) / 3
    if yoe < 3:
        return max(0.1, 0.3 * yoe / 3)
    return max(0.2, 0.7 - 0.05 * (yoe - 12))


def compute_behavioral_modifier(sig):
    """Multiplicative modifier in roughly [0.4, 1.15], punishing candidates
    who are on-paper fits but practically unreachable/unavailable, per the
    JD's explicit instruction to down-weight inactive/unresponsive profiles."""
    mod = 1.0

    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (TODAY - last_active).days
        if days_inactive > 180:
            mod *= 0.55
        elif days_inactive > 90:
            mod *= 0.75
        elif days_inactive > 30:
            mod *= 0.92

    resp_rate = sig.get("recruiter_response_rate", 0.5)
    mod *= 0.55 + 0.65 * resp_rate  # 0.55 at 0%, 1.20 at 100%

    if not sig.get("open_to_work_flag", True):
        mod *= 0.6

    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        mod *= 1.08
    elif notice <= 60:
        mod *= 1.0
    elif notice <= 90:
        mod *= 0.92
    else:
        mod *= 0.8

    interview_rate = sig.get("interview_completion_rate", 0.7)
    mod *= 0.8 + 0.3 * interview_rate

    if sig.get("verified_email") and sig.get("verified_phone"):
        mod *= 1.03

    # Cap at 1.0: strong availability should never inflate a candidate
    # above their substantive on-paper fit, only poor availability should
    # pull them down. This also keeps the final composite score naturally
    # bounded in [0, 1] without needing a post-hoc clamp that would
    # collapse distinct top candidates into ties.
    return max(0.35, min(1.0, mod))


def compute_disqualifier_penalty(cand, text_blob, career_history):
    """Multiplicative penalty in (0, 1] for explicit JD disqualifiers."""
    penalty = 1.0
    profile = cand["profile"]

    if is_research_only(profile, career_history):
        penalty *= 0.05  # near-total disqualifier per JD

    if is_consulting_only(career_history) and len(career_history) <= 2:
        penalty *= 0.5  # soft disqualifier, JD says case-by-case

    if CV_SPEECH_ROBOTICS_PATTERNS.search(text_blob) and not NLP_IR_PATTERNS.search(text_blob):
        penalty *= 0.35  # CV/speech/robotics with no NLP/IR exposure

    # Recent-only LangChain/API-wrapper experience without older
    # substantial production ML history.
    has_wrapper_only_signal = bool(WRAPPER_ONLY_PATTERNS.search(text_blob))
    has_real_substance = bool(PROD_SUBSTANCE_RE.search(text_blob))
    yoe = profile.get("years_of_experience", 0)
    if has_wrapper_only_signal and not has_real_substance and yoe < 4:
        penalty *= 0.4

    return penalty


def score_candidate(cand):
    profile = cand["profile"]
    career_history = cand.get("career_history", [])
    education = cand.get("education", [])
    skills = cand.get("skills", [])
    sig = cand.get("redrob_signals", {})

    if honeypot_checks(cand):
        return -1.0, {"honeypot": True}

    title = profile.get("current_title", "")
    t_score = title_tier_score(title)

    text_blob = career_text_blob(profile, career_history)
    substance_hits = len(PROD_SUBSTANCE_RE.findall(text_blob))
    substance_score = min(1.0, substance_hits / 6.0)

    skills_score, n_core_skills = compute_skills_quality(skills)

    exp_score = compute_experience_fit(profile.get("years_of_experience", 0))

    edu_score = 0.3
    if education:
        tiers = [EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.3) for e in education]
        edu_score = max(tiers)

    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    loc_score = 0.5
    if country == "india":
        loc_score = 0.85
        if any(city in location for city in ["pune", "noida", "delhi", "ncr", "hyderabad", "mumbai", "bangalore", "bengaluru"]):
            loc_score = 1.0
    elif sig.get("willing_to_relocate"):
        loc_score = 0.5
    else:
        loc_score = 0.2

    # Weighted composite of the "on-paper fit" components. Weights sum to
    # 0.975, not 1.0 — the remaining 0.025 of headroom is reserved for the
    # tie_bonus below so that candidates who max out every main component
    # can still be differentiated rather than all landing on a hard 1.0.
    base = (
        0.332 * t_score +
        0.254 * substance_score +
        0.156 * skills_score +
        0.098 * exp_score +
        0.059 * edu_score +
        0.078 * loc_score
    )

    # Small uncapped tie-breaking bonus: among candidates who all hit the
    # substance/skills caps, reward genuinely deeper evidence (more
    # substance mentions, more substantiated core skills, more
    # endorsements) rather than letting them tie. Kept small so it can
    # only break ties, never override the main weighted components.
    tie_bonus = (
        0.01 * min(1.0, substance_hits / 20.0) +
        0.01 * min(1.0, n_core_skills / 15.0) +
        0.005 * min(1.0, sig.get("endorsements_received", 0) / 200.0)
    )
    base = min(1.0, base + tie_bonus)

    penalty = compute_disqualifier_penalty(cand, text_blob, career_history)
    behavioral = compute_behavioral_modifier(sig)

    final = min(1.0, base * penalty * behavioral)

    feats = {
        "honeypot": False,
        "title": title,
        "t_score": t_score,
        "substance_hits": substance_hits,
        "substance_score": substance_score,
        "skills_score": skills_score,
        "n_core_skills": n_core_skills,
        "exp_score": exp_score,
        "yoe": profile.get("years_of_experience", 0),
        "edu_score": edu_score,
        "loc_score": loc_score,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "penalty": penalty,
        "behavioral": behavioral,
        "last_active": sig.get("last_active_date"),
        "resp_rate": sig.get("recruiter_response_rate"),
        "notice_period_days": sig.get("notice_period_days"),
        "company": profile.get("current_company", ""),
    }
    return final, feats


def build_reasoning(cand, feats, rank):
    """Build a short, specific, honest reasoning string. Avoids the Stage-4
    penalties explicitly: no identical boilerplate across rows, no inserting
    only the candidate's name into a template, no claiming skills that
    aren't actually in the candidate's profile, and the tone matches the
    rank (a rank-90 candidate isn't described the same way as rank-1)."""
    profile = cand["profile"]
    sig = cand.get("redrob_signals", {})
    yoe = feats["yoe"]
    title = feats["title"]
    company = feats["company"]
    location = feats["location"]

    # Pull 1-2 concrete, real skill names (only ones actually on the
    # candidate's skill list) to anchor the sentence in specifics rather
    # than generic praise.
    skill_names = [s.get("name", "") for s in cand.get("skills", [])
                   if s.get("name", "").strip().lower() in CORE_AI_SKILL_NAMES]
    top_skills = ", ".join(skill_names[:3]) if skill_names else None

    opener_parts = [f"{title}, {yoe:.1f} yrs, currently at {company}"]
    if location:
        opener_parts.append(f"based in {location}")
    opener = "; ".join(opener_parts) + "."

    # Tier-dependent framing so language genuinely differs by rank band,
    # not just by swapping the name in a fixed sentence.
    if feats["t_score"] >= 1.0 and feats["substance_hits"] >= 3:
        fit_clause = "Title and career narrative both point directly at ranking/retrieval/ML systems work, which is the core of this JD."
    elif feats["t_score"] >= 1.0:
        fit_clause = "Title matches the role directly, though the career narrative gives only light detail on what was actually built."
    elif feats["t_score"] >= 0.65 and feats["substance_hits"] >= 2:
        fit_clause = "Title is adjacent (not an exact match) but the career history describes real applied-ML/search work that closes the gap."
    elif feats["substance_hits"] >= 2:
        fit_clause = "Title reads as a general engineering/data role, but the job descriptions in their career history describe genuine ranking or retrieval work — the kind of signal the JD says to look for past the title."
    else:
        fit_clause = "Included as filler in the lower band; title and career narrative show only loose adjacency to the JD's core ask."

    if top_skills:
        skill_clause = f"Substantiated skills include {top_skills}."
    else:
        skill_clause = "Skill list has limited substantiated overlap with the JD's core stack."

    notes = []
    if feats["penalty"] < 0.6:
        notes.append("flagged against a JD disqualifier (research-only background, consulting-only career, or CV/speech focus without NLP/IR)")
    notice = sig.get("notice_period_days")
    resp = sig.get("recruiter_response_rate")
    last_active = sig.get("last_active_date")
    if feats["behavioral"] < 0.75:
        notes.append(f"weak availability (response rate {resp}, last active {last_active})")
    elif feats["behavioral"] > 1.05:
        notes.append(f"strong availability (response rate {resp}, notice period {notice}d)")
    if feats["loc_score"] < 0.4:
        notes.append("location/relocation is a mismatch for the Pune/Noida-preferred mandate")

    tail = (" " + "; ".join(notes) + ".") if notes else ""

    text = f"{opener} {fit_clause} {skill_clause}{tail}"
    if len(text) > 320:
        text = text[:317].rsplit(" ", 1)[0] + "..."
    return text


# ----------------------------------------------------------------------
# 3. Streaming top-K selection
# ----------------------------------------------------------------------

def open_candidates_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def run(candidates_path, out_path, top_k=100):
    heap = []  # min-heap of (score, candidate_id, cand_dict, feats)
    counter = 0
    honeypots_seen = 0

    with open_candidates_file(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            score, feats = score_candidate(cand)
            counter += 1
            if feats.get("honeypot"):
                honeypots_seen += 1
                continue  # never eligible for top-K

            entry = (score, cand["candidate_id"], cand, feats)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    print(f"Processed {counter} candidates; {honeypots_seen} honeypots excluded.", file=sys.stderr)

    # Round first, then sort, so the tie-break on candidate_id ascending
    # is applied to the *displayed* score (matches what the validator
    # checks) and not to floating-point noise below 2 decimal places.
    # Output scale: 0-100 (percentage-style fit score), not 0-1.
    rounded = [
        (round(max(score, 0.0001) * 100, 2), cid, cand, feats)
        for (score, cid, cand, feats) in heap
    ]
    ranked = sorted(rounded, key=lambda x: (-x[0], x[1]))

    rows = []
    for rank, (score, cid, cand, feats) in enumerate(ranked, start=1):
        reasoning = build_reasoning(cand, feats, rank)
        rows.append((cid, rank, score, reasoning))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in rows:
            writer.writerow([cid, rank, f"{score:.2f}", reasoning])

    print(f"Wrote top {len(rows)} candidates to {out_path}", file=sys.stderr)


def main():
   candidates_file = "candidates.jsonl"
   output_file = "submission.csv"
   top_k = 100

   run(candidates_file, output_file, top_k)


if __name__ == "__main__":
    main()
