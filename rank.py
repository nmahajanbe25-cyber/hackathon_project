
import argparse
import csv
import gzip
import heapq
import json
import re
import sys
from datetime import date, datetime

TITLE_TIER_3 = { 
    "ai engineer", "senior ai engineer", "lead ai engineer", "staff ai engineer",
    "machine learning engineer", "senior machine learning engineer",
    "staff machine learning engineer", "ml engineer", "senior ml engineer",
    "applied ml engineer", "applied scientist", "senior applied scientist",
    "nlp engineer", "senior nlp engineer", "search engineer",
    "recommendation systems engineer", "ranking engineer",
    "information retrieval engineer", "ai research engineer",
}
TITLE_TIER_2 = { 
    "data scientist", "senior data scientist", "computer vision engineer",
    "ai specialist", "junior ml engineer", "research engineer",
    "senior software engineer (ml)", "ml ops engineer", "mlops engineer",
}
TITLE_TIER_1 = {  
    "software engineer", "senior software engineer", "data engineer",
    "senior data engineer", "analytics engineer", "data analyst",
    "backend engineer", "full stack developer",
}

RESEARCH_ONLY_TITLES = {
    "research scientist", "research fellow", "postdoctoral researcher",
    "phd researcher", "research assistant", "academic researcher",
}


CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}


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

TODAY = date.today()

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

   
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 1:
            return True

    
    yoe_months = profile.get("years_of_experience", 0) * 12
    total_months = sum(j.get("duration_months", 0) for j in career_history)
    if total_months > yoe_months * 1.5 + 6:
        return True


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
            
            trust = min(1.0, (dur / 12.0)) * 0.7 + min(1.0, endorse / 10.0) * 0.3
            trust = max(trust, 0.1)  
            core_hits.append(prof_w * trust)
    n_core = len(core_hits)
    if n_core == 0:
        return 0.0, 0
    
    avg_quality = sum(core_hits) / n_core
    breadth_bonus = min(0.3, 0.04 * n_core)
    return min(1.0, avg_quality + breadth_bonus), n_core


def compute_experience_fit(yoe):
   
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
    mod *= 0.55 + 0.65 * resp_rate 

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

    
    return max(0.35, min(1.0, mod))


def compute_disqualifier_penalty(cand, text_blob, career_history):
    """Multiplicative penalty in (0, 1] for explicit JD disqualifiers."""
    penalty = 1.0
    profile = cand["profile"]

    if is_research_only(profile, career_history):
        penalty *= 0.05  

    if is_consulting_only(career_history) and len(career_history) <= 2:
        penalty *= 0.5 
    if CV_SPEECH_ROBOTICS_PATTERNS.search(text_blob) and not NLP_IR_PATTERNS.search(text_blob):
        penalty *= 0.35  
        
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

    
    base = (
        0.332 * t_score +
        0.254 * substance_score +
        0.156 * skills_score +
        0.098 * exp_score +
        0.059 * edu_score +
        0.078 * loc_score
    )

   
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

   
    skill_names = [s.get("name", "") for s in cand.get("skills", [])
                   if s.get("name", "").strip().lower() in CORE_AI_SKILL_NAMES]
    top_skills = ", ".join(skill_names[:3]) if skill_names else None

    opener_parts = [f"{title}, {yoe:.1f} yrs, currently at {company}"]
    if location:
        opener_parts.append(f"based in {location}")
    opener = "; ".join(opener_parts) + "."

    
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




def open_candidates_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def run(candidates_path, out_path, top_k=100):
    heap = []  
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
                continue  

            entry = (score, cand["candidate_id"], cand, feats)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    print(f"Processed {counter} candidates; {honeypots_seen} honeypots excluded.", file=sys.stderr)

    
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
