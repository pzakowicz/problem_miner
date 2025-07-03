import os
import sqlite3
import requests
import requests.auth
import re
import time
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from threading import Lock
import io, csv
import threading
import praw
from praw.models import MoreComments
from psaw import PushshiftAPI

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv('REDDIT_CLIENT_ID') or ''
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET') or ''
USER_AGENT = os.getenv('REDDIT_USER_AGENT') or ''
USERNAME = os.getenv('REDDIT_USERNAME') or ''
PASSWORD = os.getenv('REDDIT_PASSWORD') or ''
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

DB_PATH = 'reddit_pain_points.db'

PAIN_KEYWORDS = [
    'problem', 'struggle', 'issue', "can't", 'hate', 'wish', 'barrier', 'challenge'
]

# Add OpenAI import
try:
    import openai
except ImportError:
    openai = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'static' directory at /static (not /)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

class ScanRequest(BaseModel):
    subreddit: str
    batch_size: int = 20
    total_posts: int = 50

class LabelRequest(BaseModel):
    subreddit: str
    problem_id: str
    problem_text: str
    context: str = ''
    label: str
    notes: str = ''

class SubredditDeleteRequest(BaseModel):
    subreddit: str

class RecategorizeRequest(BaseModel):
    subreddit: str

# Global scan progress tracker
scan_progress = {}
scan_progress_lock = Lock()

# Global deep scan progress tracker
deep_scan_progress = {}
deep_scan_progress_lock = Lock()

# Global deep scan stop flags
deep_scan_stop_flags = {}

recategorize_locks = {}
recategorize_progress = {}

export_progress = {}
export_results = {}
export_locks = {}

# Helper: Get OAuth token
def get_token():
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {
        'grant_type': 'password',
        'username': USERNAME,
        'password': PASSWORD
    }
    headers = {'User-Agent': USER_AGENT}
    res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
    res.raise_for_status()
    return res.json()['access_token']

# Helper: Create subreddit-specific table
def create_table(conn, table):
    with conn:
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subreddit TEXT,
                post_title TEXT,
                post_id TEXT,
                comment_id TEXT UNIQUE,
                author TEXT,
                problem TEXT,
                context TEXT,
                category TEXT,
                upvotes INTEGER,
                created_at TEXT,
                comment_url TEXT
            )
        ''')

# Helper: Save a matched problem to DB (deduplication via UNIQUE)
def save_problem(conn, table, data):
    print(f"[DeepScan][DEBUG] Attempting to insert: {data}")
    with conn:
        try:
            conn.execute(f'''
                INSERT INTO {table} (subreddit, post_title, post_id, comment_id, author, problem, context, category, upvotes, created_at, comment_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                data['subreddit'],
                data['post_title'],
                data['post_id'],
                data['comment_id'],
                data['author'],
                data['problem'],
                data['context'],
                data['category'],
                data['upvotes'],
                data['created_at'],
                data['comment_url']
            ))
            print(f"[DeepScan] Saved problem for comment {data['comment_id']}: {data['problem']}")
        except Exception as e:
            print(f"[DeepScan] ERROR saving problem: {e}")

# Helper: Recursively extract all comments (including nested)
def extract_comments(comment_list, results):
    for comment in comment_list:
        if comment['kind'] != 't1':
            continue
        c = comment['data']
        results.append(c)
        if c.get('replies') and isinstance(c['replies'], dict):
            extract_comments(c['replies']['data']['children'], results)

def extract_problem_sentences(comment_body, keywords):
    sentences = re.split(r'(?<=[.!?]) +', comment_body)
    return [s.strip() for s in sentences if any(kw in s.lower() for kw in keywords)]

def extract_problem_statement_openai(comment_body):
    """
    Uses OpenAI to extract a concise, quoted problem statement from a comment.
    Returns the quoted sentence(s) or None if no problem is found.
    """
    if not openai or not OPENAI_API_KEY or not comment_body.strip():
        return None
    openai.api_key = OPENAI_API_KEY
    prompt = (
        "Does the following text describe a real problem, pain point, or challenge? "
        "If yes, quote the sentence or two (exactly as written) that best describe the problem. "
        "If not, reply 'NO PROBLEM'.\n\n"
        f"Text: '''{comment_body}'''")
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at identifying real problems in online discussions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0
        )
        result = response.choices[0].message.content.strip()  # type: ignore
        if result.upper() == 'NO PROBLEM' or not result:
            return None
        return result
    except Exception as e:
        print(f"OpenAI API error (problem extraction): {e}")
        return None

def make_comment_url(subreddit, post_id, comment_id):
    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/_/{comment_id}"

def categorize_problem(problem):
    # Simple keyword-based fallback
    s = problem.lower()
    if any(word in s for word in ["injury", "pain", "back", "achilles", "calf", "health issues"]):
        return "injury"
    if any(word in s for word in ["consistency", "routine", "habit"]):
        return "consistency"
    if any(word in s for word in ["progress", "progression", "stuck", "plateau", "slow"]):
        return "progress"
    return "other"

def categorize_problem_openai(problem, subreddit):
    if not openai or not OPENAI_API_KEY:
        return None
    openai.api_key = OPENAI_API_KEY
    prompt = (
        f"You are an expert at organizing and categorizing online discussions. For the subreddit '{subreddit}', imagine a set of at most 10 mutually exclusive categories that best capture the main types of pain points or problems discussed in this community. "
        f"Given the following problem, assign it to the single most appropriate category from your set. Respond with only the category name.\n\n"
        f"Problem: \"{problem}\"\nCategory:"
    )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes subreddit pain points."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0
        )
        category = response.choices[0].message.content.strip()  # type: ignore
        return category.split('\n')[0].split(',')[0].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# --- CATEGORY GENERATION AND ASSIGNMENT ---
def get_or_generate_categories(subreddit, sample_problems):
    if not openai or not OPENAI_API_KEY:
        return [f"Category {i+1}" for i in range(10)]
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cat_table = f"categories_{subreddit}"
    # Check if categories already exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (cat_table,))
    if cursor.fetchone():
        cursor.execute(f"SELECT category FROM {cat_table}")
        cats = [row[0] for row in cursor.fetchall()]
        conn.close()
        return cats
    # Generate categories using OpenAI
    openai.api_key = OPENAI_API_KEY
    sample_text = '\n'.join(f'- {p}' for p in sample_problems[:30])
    prompt = (
        f"Given these examples of problems from r/{subreddit}:\n{sample_text}\n\n"
        "Generate a list of 10 mutually exclusive, broad categories that best capture the main types of pain points discussed. "
        "Respond with only the category names as a comma-separated list."
    )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are an expert at organizing and categorizing online discussions."},
                      {"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0
        )
        cats = response.choices[0].message.content.strip().split(',')  # type: ignore
        cats = [c.strip() for c in cats if c.strip()][:10]
    except Exception as e:
        print(f"OpenAI API error (category gen): {e}")
        cats = [f"Category {i+1}" for i in range(10)]
    # Store categories
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {cat_table} (category TEXT PRIMARY KEY)")
    for c in cats:
        cursor.execute(f"INSERT OR IGNORE INTO {cat_table} (category) VALUES (?)", (c,))
    conn.commit()
    conn.close()
    return cats

def assign_category_openai(problem, categories, subreddit):
    if not openai or not OPENAI_API_KEY:
        return categories[0]
    openai.api_key = OPENAI_API_KEY
    cat_list = ', '.join(categories)
    prompt = (
        f"Given the following problem from r/{subreddit}:\n'{problem}'\n\n"
        f"Assign it to the single most appropriate category from this list: {cat_list}.\n"
        "Respond with only the category name."
    )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are a helpful assistant that assigns problems to categories."},
                      {"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0
        )
        cat = response.choices[0].message.content.strip()  # type: ignore
        # Fuzzy match to closest category
        for c in categories:
            if cat.lower() == c.lower():
                return c
        # If not exact, pick the closest by substring
        for c in categories:
            if cat.lower() in c.lower() or c.lower() in cat.lower():
                return c
        return categories[0]  # fallback
    except Exception as e:
        print(f"OpenAI API error (assign cat): {e}")
        return categories[0]

def extract_and_categorize_problem_openai(text, categories=None, post_title=None):
    """
    Uses OpenAI to extract a quoted problem statement and assign a category in a single call.
    The extracted problem must include any necessary context inside the quoted text.
    Returns (problem, context, category) or (None, None, None) if no valid problem is found.
    """
    if not openai or not OPENAI_API_KEY or not text.strip():
        return None, None, None
    openai.api_key = OPENAI_API_KEY
    context_intro = f"Post title: {post_title}\n" if post_title else ""
    if categories:
        cat_list = ', '.join(categories)
        prompt = (
            f"{context_intro}Comment or post body: {text}\n\n"
            "Does the above text contain a clear, personal problem, pain point, or challenge directly related to the topic? "
            "Only accept a real need, pain, struggle, question, or thing the user does not know how to solve. "
            "Reject general advice, replies to others, vague statements, or statements that only describe goals or achievements without mentioning any difficulty, pain, or obstacle. "
            "Reject text that is too short to understand without extra context. "
            "If valid, quote exactly the full sentence or 2–3 consecutive sentences that clearly show the problem, including any context needed, all inside PROBLEM. "
            f"Assign it to one of these categories: {cat_list}.\n"
            "Reply only in this format:\nPROBLEM: \"quoted problem with context\"\nCATEGORY: category_name\n"
            "If no valid personal problem exists, reply 'NO PROBLEM'."
        )
    else:
        prompt = (
            f"{context_intro}Comment or post body: {text}\n\n"
            "Does the above text contain a clear, personal problem, pain point, or challenge directly related to the topic? "
            "Only accept a real need, pain, struggle, question, or thing the user does not know how to solve. "
            "Reject general advice, replies to others, vague statements, or statements that only describe goals or achievements without mentioning any difficulty, pain, or obstacle. "
            "Reject text that is too short to understand without extra context. "
            "If valid, quote exactly the full sentence or 2–3 consecutive sentences that clearly show the problem, including any context needed, all inside PROBLEM. "
            "Reply only in this format:\nPROBLEM: \"quoted problem with context\"\n"
            "If no valid personal problem exists, reply 'NO PROBLEM'."
        )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at detecting clear, personal problems in online discussions. Only extract genuine user-expressed needs, struggles, or unanswered questions. Never accept goals or general statements alone."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0
        )
        result = response.choices[0].message.content.strip()  # type: ignore
        if result.upper() == 'NO PROBLEM' or not result:
            return None, None, None
        # Parse response
        problem = None
        category = None
        for line in result.split('\n'):
            if line.strip().startswith('PROBLEM:'):
                problem = line.split('PROBLEM:', 1)[1].strip().strip('"')
            elif line.strip().startswith('CATEGORY:'):
                category = line.split('CATEGORY:', 1)[1].strip()
        # Fallback: try to find quoted text for problem
        if not problem:
            import re
            m = re.search(r'"([^"]+)"', result)
            if m:
                problem = m.group(1)
        return problem, None, category
    except Exception as e:
        print(f"OpenAI API error (extract & categorize): {e}")
        return None, None, None

@app.get("/scan_progress")
def get_scan_progress(subreddit: str):
    key = subreddit.lower()
    with scan_progress_lock:
        prog = scan_progress.get(key, None)
    if prog is None:
        return {"status": "idle"}
    return prog

@app.get("/deep_scan_progress")
def get_deep_scan_progress(subreddit: str = "", start_date: str = ""):
    key = f"{subreddit.lower()}_{start_date}"
    with deep_scan_progress_lock:
        prog = deep_scan_progress.get(key, None)
        if not prog:
            return {"status": "idle"}
        return prog

def migrate_add_context_column():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pain_points_%'")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if 'context' not in columns:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN context TEXT")
                print(f"Added 'context' column to {table}")
            except Exception as e:
                print(f"Migration error for {table}: {e}")
    conn.commit()
    # Verify all tables have the context column
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        # print(f"Schema for {table}: {columns}")  # Debug: schema output (now disabled)
        if 'context' not in columns:
            raise RuntimeError(f"Table {table} is missing 'context' column after migration!")
    conn.close()

def migrate_add_labels_table():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS problem_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subreddit TEXT,
            problem_id TEXT,
            problem_text TEXT,
            context TEXT,
            label TEXT,
            notes TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call migration at startup
migrate_add_context_column()
migrate_add_labels_table()

@app.post("/scan")
def scan_subreddit(req: ScanRequest):
    migrate_add_context_column()
    subreddit = str(req.subreddit or '').strip()
    batch_size = req.batch_size
    total_posts = req.total_posts
    table = f"pain_points_{subreddit.lower()}"
    token = get_token()
    headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}
    # Close and reopen connection
    conn = sqlite3.connect(DB_PATH, timeout=30)
    create_table(conn, table)
    n_posts = 0
    n_problems = 0
    after = None
    sample_problems = []
    sample_limit = 30
    categories = None
    sample_collected = False
    # Initialize progress
    key = subreddit.lower()
    with scan_progress_lock:
        scan_progress[key] = {"status": "scanning", "posts_scanned": 0, "total_posts": total_posts, "problems_found": 0}
    try:
        while n_posts < total_posts:
            url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={batch_size}"
            if after:
                url += f"&after={after}"
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()['data']
            posts = data['children']
            after = data.get('after')
            if not posts:
                break
            for post in posts:
                n_posts += 1
                if n_posts > total_posts:
                    break
                # Update progress (increment only)
                with scan_progress_lock:
                    scan_progress[key]["posts_scanned"] = max(scan_progress[key]["posts_scanned"], n_posts)
                post_data = post['data']
                post_id = post_data['id']
                post_title = post_data.get('title', '')
                # --- Skip if post already scanned ---
                cursor = conn.cursor()
                cursor.execute(f"SELECT 1 FROM {table} WHERE comment_id=?", (post_id,))
                if cursor.fetchone():
                    continue
                # --- Scan the original post body ---
                post_body = post_data.get('selftext', '')
                if post_body.strip():
                    if not categories and len(sample_problems) < sample_limit:
                        problem, context, _ = extract_and_categorize_problem_openai(post_body, None, post_title)
                        if problem:
                            sample_problems.append(problem)
                            if len(sample_problems) >= sample_limit:
                                categories = get_or_generate_categories(subreddit, sample_problems)
                    else:
                        problem, context, category = extract_and_categorize_problem_openai(post_body, categories, post_title)
                        if problem and category:
                            n_problems += 1
                            with scan_progress_lock:
                                scan_progress[key]["problems_found"] = max(scan_progress[key]["problems_found"], n_problems)
                            data = {
                                'subreddit': subreddit,
                                'post_title': post_title,
                                'post_id': post_id,
                                'comment_id': post_id,  # Use post_id as comment_id for original post
                                'author': post_data.get('author'),
                                'problem': problem,
                                'context': context,
                                'category': category,
                                'upvotes': post_data.get('score', 0),
                                'created_at': datetime.utcfromtimestamp(post_data.get('created_utc', 0)).isoformat() if post_data.get('created_utc') else None,
                                'comment_url': make_comment_url(subreddit, post_id, post_id)
                            }
                            save_problem(conn, table, data)
                # --- Scan all comments ---
                comments_url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}?limit=500"
                c_resp = requests.get(comments_url, headers=headers)
                c_resp.raise_for_status()
                comment_listing = c_resp.json()
                if len(comment_listing) < 2:
                    continue
                all_comments = []
                extract_comments(comment_listing[1]['data']['children'], all_comments)
                for c in all_comments:
                    body = c.get('body', '')
                    comment_id = c.get('id')
                    if not body.strip():
                        continue
                    cursor.execute(f"SELECT 1 FROM {table} WHERE comment_id=?", (comment_id,))
                    if cursor.fetchone():
                        continue  # Already scanned
                    if not categories and len(sample_problems) < sample_limit:
                        problem, context, _ = extract_and_categorize_problem_openai(body, None, post_title)
                        if problem:
                            sample_problems.append(problem)
                            if len(sample_problems) >= sample_limit:
                                categories = get_or_generate_categories(subreddit, sample_problems)
                    else:
                        problem, context, category = extract_and_categorize_problem_openai(body, categories, post_title)
                        if problem and category:
                            n_problems += 1
                            with scan_progress_lock:
                                scan_progress[key]["problems_found"] = max(scan_progress[key]["problems_found"], n_problems)
                            data = {
                                'subreddit': subreddit,
                                'post_title': post_title,
                                'post_id': post_id,
                                'comment_id': comment_id,
                                'author': c.get('author'),
                                'problem': problem,
                                'context': context,
                                'category': category,
                                'upvotes': c.get('score', 0),
                                'created_at': datetime.utcfromtimestamp(c.get('created_utc', 0)).isoformat() if c.get('created_utc') else None,
                                'comment_url': make_comment_url(subreddit, post_id, comment_id)
                            }
                            save_problem(conn, table, data)
                    
                time.sleep(1)
            if not after:
                break
        # If categories were never generated (not enough samples), generate with what we have
        if not categories:
            categories = get_or_generate_categories(subreddit, sample_problems)
        remove_non_good_and_deduplicate(conn, table, subreddit)
    finally:
        with scan_progress_lock:
            scan_progress[key]["status"] = "done"
        conn.close()
    return {"message": f"✅ Scan complete: {n_problems} problems added from {n_posts} posts."}

@app.get("/summary")
def get_summary(subreddit: str = Query(None)):
    migrate_add_context_column()
    # Reopen connection after migration
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    all_rows = []
    subr = str(subreddit or '')
    if subr:
        # Only aggregate from the table for the selected subreddit
        table_name = f"pain_points_{subr}"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            # Join with problem_labels to get label for each problem
            cursor.execute(f'''
                SELECT p.category, p.problem, p.context, p.post_title, p.upvotes, p.author, p.comment_url, p.subreddit, l.label
                FROM {table_name} p
                LEFT JOIN problem_labels l ON p.comment_id = l.problem_id AND l.subreddit = ?
                WHERE l.label IS NULL OR l.label != 'bad'
            ''', (subr,))
            all_rows.extend(cursor.fetchall())
    else:
        # No subreddit selected, return empty list
        conn.close()
        return []
    summary = {}
    for cat, problem, context, post_title, upvotes, author, url, subreddit, label in all_rows:
        if not cat:
            cat = 'Uncategorized'
        if cat not in summary:
            summary[cat] = {"category": cat, "total_upvotes": 0, "problems": []}
        summary[cat]["total_upvotes"] += upvotes
        summary[cat]["problems"].append({
            "problem": problem,
            "context": context,
            "post_title": post_title,
            "upvotes": upvotes,
            "author": author,
            "comment_url": url,
            "subreddit": subreddit,
            "label": label
        })
    # Sort categories by total_upvotes, then by number of problems
    sorted_summary = sorted(
        summary.values(),
        key=lambda x: (-x["total_upvotes"], -len(x["problems"]))
    )
    conn.close()
    return sorted_summary

@app.post("/clear")
def clear_data():
    # Do NOT drop any tables, just return a success message
    return JSONResponse({"message": "Frontend data cleared (no tables dropped)."})

@app.get("/subreddits")
def list_subreddits():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pain_points_%'")
    tables = [row[0] for row in cursor.fetchall()]
    # Remove the 'pain_points_' prefix
    subreddits = [t[len('pain_points_'):] for t in tables]
    conn.close()
    return subreddits

@app.post("/recategorize")
def recategorize_subreddit(req: RecategorizeRequest):
    subr = str(req.subreddit or '').strip().lower()
    if recategorize_locks.get(subr):
        return JSONResponse({"error": "Recategorization already in progress."}, status_code=429)
    recategorize_locks[subr] = True
    recategorize_progress[subr] = {"status": "running", "total": 0, "done": 0, "message": ""}
    try:
        table = f"pain_points_{subr}"
        cat_table = f"categories_{subr}"
        conn = sqlite3.connect(DB_PATH, timeout=60)
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, problem FROM {table}")
        rows = cursor.fetchall()
        problems = [row[1] for row in rows if row[1]]
        ids = [row[0] for row in rows if row[1]]
        total = len(problems)
        recategorize_progress[subr]["total"] = total
        if not problems:
            conn.close()
            recategorize_progress[subr] = {"status": "done", "total": 0, "done": 0, "message": "No problems to recategorize."}
            return {"message": "No problems to recategorize."}
        categories = get_or_generate_categories(subr, problems)
        new_cats = []
        for i, prob in enumerate(problems):
            new_cat = assign_category_openai(prob, categories, subr)
            new_cats.append(new_cat)
            recategorize_progress[subr]["done"] = i + 1
        try:
            cursor.execute("BEGIN TRANSACTION")
            for row_id, new_cat in zip(ids, new_cats):
                cursor.execute(f"UPDATE {table} SET category=? WHERE id=?", (new_cat, row_id))
            conn.commit()
        finally:
            conn.close()
        recategorize_progress[subr] = {"status": "done", "total": total, "done": total, "message": f"Recategorized {total} problems with {len(categories)} categories."}
        return {"message": f"Recategorized {total} problems with {len(categories)} categories."}
    except Exception as e:
        recategorize_progress[subr] = {"status": "error", "total": recategorize_progress[subr].get("total", 0), "done": recategorize_progress[subr].get("done", 0), "message": str(e)}
        raise
    finally:
        recategorize_locks[subr] = False

@app.get("/recategorize_progress")
def get_recategorize_progress(subreddit: str):
    subr = subreddit.strip().lower()
    return recategorize_progress.get(subr, {"status": "idle", "total": 0, "done": 0, "message": ""})

@app.get("/problems_for_labeling")
def get_problems_for_labeling(subreddit: str = "", limit: int = 50):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    subr = str(subreddit or '')
    if subr:
        table = f"pain_points_{subr}"
        cursor.execute(f"SELECT comment_id, problem, context, post_title, upvotes, author, comment_url FROM {table}")
        problems = cursor.fetchall()
        # Exclude all problems that have any label
        cursor.execute("SELECT problem_id FROM problem_labels WHERE subreddit=?", (subr,))
        labeled_any = set(row[0] for row in cursor.fetchall())
        result = [dict(comment_id=row[0], problem=row[1], context=row[2], post_title=row[3], upvotes=row[4], author=row[5], comment_url=row[6])
                  for row in problems if row[0] not in labeled_any]
    else:
        result = []
    conn.close()
    return result[:limit]

@app.post("/label_problem")
def label_problem(data: LabelRequest):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO problem_labels (subreddit, problem_id, problem_text, context, label, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    ''', (
        str(data.subreddit),
        str(data.problem_id),
        str(data.problem_text),
        str(data.context or ''),
        str(data.label),
        str(data.notes or '')
    ))
    conn.commit()
    conn.close()
    return {"status": "ok"}

# On scan: after scan, delete all non-'good' labeled problems for the subreddit, and do not add duplicates (by comment_id/post_id)
def remove_non_good_and_deduplicate(conn, table, subreddit):
    cursor = conn.cursor()
    subr = subreddit if isinstance(subreddit, str) and subreddit is not None else ''
    # Get all problem_ids labeled as 'bad' or 'uncertain'
    cursor.execute("SELECT problem_id FROM problem_labels WHERE subreddit=? AND (label='bad' OR label='uncertain')", (subr,))
    bad_ids = set(str(row[0]) for row in cursor.fetchall() if row[0] is not None)
    if bad_ids:
        placeholders = ','.join('?' for _ in bad_ids)
        cursor.execute(f"DELETE FROM {table} WHERE comment_id IN ({placeholders})", tuple(bad_ids))
        conn.commit()
    # Do not delete unlabeled or 'good' problems

# Serve index.html at the root path
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/delete_subreddit")
def delete_subreddit(req: SubredditDeleteRequest):
    subr = str(req.subreddit or '').strip().lower()
    if not subr:
        raise HTTPException(status_code=400, detail="No subreddit provided.")
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 5000")  # Wait up to 5 seconds for locks
    cursor = conn.cursor()
    pain_table = f"pain_points_{subr}"
    cat_table = f"categories_{subr}"
    # Drop pain points table if exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (pain_table,))
    pain_exists = cursor.fetchone() is not None
    if pain_exists:
        cursor.execute(f"DROP TABLE IF EXISTS {pain_table}")
    # Drop categories table if exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (cat_table,))
    cat_exists = cursor.fetchone() is not None
    if cat_exists:
        cursor.execute(f"DROP TABLE IF EXISTS {cat_table}")
    conn.commit()
    conn.close()
    return {"message": f"Deleted tables: {', '.join([t for t, e in [(pain_table, pain_exists), (cat_table, cat_exists)] if e]) or 'None'} for subreddit '{subr}'."}

@app.post("/clear_labels")
def clear_labels():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM problem_labels")
    conn.commit()
    conn.close()
    return {"message": "All label data cleared from problem_labels table."}

@app.post("/deep_scan")
def deep_scan(data: dict, background_tasks: BackgroundTasks):
    subreddit = data.get("subreddit", "").strip()
    start_date = data.get("start_date", "").strip()
    if not subreddit or not start_date:
        return {"message": "Missing subreddit or start_date."}
    # Start the background scan task (to be implemented)
    background_tasks.add_task(deep_scan_task, subreddit, start_date)
    return {"message": f"Started deep scan for r/{subreddit} since {start_date}."}

@app.post("/stop_deep_scan")
def stop_deep_scan(data: dict):
    subreddit = data.get("subreddit", "").strip()
    start_date = data.get("start_date", "").strip()
    key = f"{subreddit.lower()}_{start_date}"
    deep_scan_stop_flags[key] = True
    print(f"[DeepScan] Stop requested for {key}")
    return {"message": f"Stop requested for deep scan of r/{subreddit} since {start_date}."}

def deep_scan_task(subreddit, start_date):
    import requests, time
    from datetime import datetime, timezone
    import openai
    import sqlite3
    import os
    BATCH_SIZE = 8
    MODEL = "gpt-3.5-turbo"
    DB_PATH = "reddit_pain_points.db"
    key = f"{subreddit.lower()}_{start_date}"
    print(f"[DeepScan] Starting deep scan for r/{subreddit} since {start_date}")
    with deep_scan_progress_lock:
        deep_scan_progress[key] = {"status": "scanning", "total_posts": 0, "posts_scanned": 0, "current_date": start_date}
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception as e:
        print(f"[DeepScan] Invalid start_date: {start_date}")
        with deep_scan_progress_lock:
            deep_scan_progress[key]["status"] = "error"
            deep_scan_progress[key]["error"] = f"Invalid start_date: {start_date}"
        return
    CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    USERNAME = os.getenv('REDDIT_USERNAME')
    PASSWORD = os.getenv('REDDIT_PASSWORD')
    def get_token():
        auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
        data = {
            'grant_type': 'password',
            'username': USERNAME,
            'password': PASSWORD
        }
        headers = {'User-Agent': USER_AGENT}
        res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
        res.raise_for_status()
        return res.json()['access_token']
    token = get_token()
    headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}
    def fetch_posts(subreddit):
        nonlocal token, headers
        url = f"https://oauth.reddit.com/r/{subreddit}/new?limit=100"
        after = None
        total = 0
        while True:
            if deep_scan_stop_flags.get(key):
                print(f"[DeepScan] Stop flag detected for {key}, exiting fetch_posts loop.")
                break
            full_url = url + (f"&after={after}" if after else "")
            print(f"[DeepScan] Fetching posts from: {full_url}")
            resp = requests.get(full_url, headers=headers)
            if resp.status_code == 401:
                print("[DeepScan] Token expired, refreshing...")
                token = get_token()
                headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}
                resp = requests.get(full_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()['data']
            posts = data['children']
            if not posts:
                break
            for post in posts:
                post_data = post['data']
                post_created = post_data.get('created_utc', 0)
                if post_created < start_ts:
                    print(f"[DeepScan] Post {post_data.get('id')} is older than start_date, stopping.")
                    return
                yield post_data
                total += 1
            after = data.get('after')
            if not after:
                break
            time.sleep(1)
        with deep_scan_progress_lock:
            deep_scan_progress[key]["total_posts"] = total
    def fetch_comments(post_id):
        nonlocal token, headers
        url = f"https://oauth.reddit.com/comments/{post_id}?limit=500"
        print(f"[DeepScan] Fetching comments for post: {post_id}")
        resp = requests.get(url, headers=headers)
        if resp.status_code == 401:
            print("[DeepScan] Token expired, refreshing...")
            token = get_token()
            headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}
            resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        comment_listing = resp.json()
        if len(comment_listing) < 2:
            return []
        all_comments = []
        def extract_comments(comment_list, results):
            for c in comment_list:
                kind = c.get('kind')
                data = c.get('data', {})
                if kind == 't1':
                    results.append(data)
                    if data.get('replies') and isinstance(data['replies'], dict):
                        extract_comments(data['replies']['data']['children'], results)
        extract_comments(comment_listing[1]['data']['children'], all_comments)
        return all_comments
    def prefilter(comment):
        body = comment.get('body', '')
        if not body or len(body) < 30:
            return False
        if body.lower().startswith('bot') or 'http' in body:
            return False
        return True
    def batch_extract_llm(batch, post_title):
        openai.api_key = OPENAI_API_KEY
        prompt = """For each Reddit comment below, extract a quoted problem statement if present, or reply NO PROBLEM.\n"""
        for i, c in enumerate(batch):
            prompt += f"{i+1}. {c['body']}\n"
        prompt += "\nRespond in this format:\n1. PROBLEM: \"...\" or NO PROBLEM\n2. ..."
        print(f"[DeepScan] Sending batch of {len(batch)} comments to OpenAI for extraction.")
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting real, personal problems from Reddit comments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256*len(batch),
                temperature=0
            )
            if (
                isinstance(response, dict)
                and 'choices' in response
                and isinstance(response['choices'], list)
                and len(response['choices']) > 0
                and 'message' in response['choices'][0]
                and 'content' in response['choices'][0]['message']
            ):
                content = response['choices'][0]['message']['content']
                print(f"[DeepScan] OpenAI response: {content[:200]}{'...' if len(content) > 200 else ''}")
                results = content.strip().split('\n')
                return results
            else:
                print(f"[DeepScan] Unexpected OpenAI response structure: {response}")
                return ["NO PROBLEM"] * len(batch)
        except Exception as e:
            print(f"[DeepScan] OpenAI API error (batch extract): {e}")
            return ["NO PROBLEM"] * len(batch)
    conn = sqlite3.connect(DB_PATH)
    table = f"pain_points_{subreddit.lower()}"
    create_table(conn, table)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    schema = cursor.fetchall()
    print(f"[DeepScan][DEBUG] Actual schema for {table}:")
    for col in schema:
        print(col)
    n_problems = 0
    n_posts = 0
    sample_problems = []
    sample_limit = 30
    categories = None
    try:
        for post in fetch_posts(subreddit):
            if deep_scan_stop_flags.get(key):
                print(f"[DeepScan] Stop flag detected for {key}, finishing current post and exiting main scan loop.")
                stop_after_this_post = True
            else:
                stop_after_this_post = False
            post_id = post['id']
            post_title = post.get('title', '')
            post_date = post.get('created_utc', 0)
            print(f"[DeepScan] Processing post: {post_id} - {post_title}")
            comments = fetch_comments(post_id)
            comments = [c for c in comments if prefilter(c)]
            print(f"[DeepScan] Post {post_id}: {len(comments)} comments after prefilter.")
            n_posts += 1
            for i in range(0, len(comments), BATCH_SIZE):
                batch = comments[i:i+BATCH_SIZE]
                results = batch_extract_llm(batch, post_title)
                for c, res in zip(batch, results):
                    if 'PROBLEM:' in res:
                        problem = res.split('PROBLEM:',1)[1].strip().strip('"')
                        # Collect sample problems for category generation
                        if not categories and len(sample_problems) < sample_limit:
                            sample_problems.append(problem)
                            if len(sample_problems) >= sample_limit:
                                categories = get_or_generate_categories(subreddit, sample_problems)
                        # Assign category if possible
                        if categories:
                            category = assign_category_openai(problem, categories, subreddit)
                        else:
                            category = 'Uncategorized'
                        data = {
                            'subreddit': subreddit,
                            'post_title': post_title,
                            'post_id': post_id,
                            'comment_id': c['id'],
                            'author': c.get('author'),
                            'problem': problem,
                            'context': '',  # Optionally extract context if needed
                            'category': category,
                            'upvotes': c.get('score', 0),
                            'created_at': datetime.utcfromtimestamp(c.get('created_utc', 0)).isoformat() if c.get('created_utc') else None,
                            'comment_url': f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/_/{c['id']}"
                        }
                        print(f"[DeepScan] Saving problem for comment {c['id']}: {problem} [Category: {category}]")
                        save_problem(conn, table, data)
                        n_problems += 1
            with deep_scan_progress_lock:
                deep_scan_progress[key]["posts_scanned"] += 1
                deep_scan_progress[key]["current_date"] = datetime.utcfromtimestamp(post_date).strftime("%Y-%m-%d") if post_date else ""
            if stop_after_this_post:
                print(f"[DeepScan] Exiting after finishing post {post_id} due to stop flag.")
                break
        summary_msg = f"✅ Deep scan complete: {n_problems} problems added from {n_posts} posts."
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            print(f"[DeepScan][DEBUG] Row count in {table} after scan: {row_count}")
        except Exception as e:
            print(f"[DeepScan][DEBUG] Error counting rows: {e}")
        with deep_scan_progress_lock:
            deep_scan_progress[key]["status"] = "done"
            deep_scan_progress[key]["summary"] = summary_msg
        print(summary_msg)
    except Exception as e:
        with deep_scan_progress_lock:
            deep_scan_progress[key]["status"] = "error"
            deep_scan_progress[key]["error"] = str(e)
        print(f"[DeepScan] ERROR: {e}")
    finally:
        conn.commit()
        conn.close()
        if key in deep_scan_stop_flags:
            del deep_scan_stop_flags[key] 

@app.get("/export_data")
def export_data(subreddit: str, start_date: str, end_date: str, limit: int = 1000000, session: Optional[str] = None):
    """Export posts and comments for a subreddit in a date range as CSV, with progress tracking."""
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        return JSONResponse({"error": "Invalid date format."}, status_code=400)
    if not session:
        session = subreddit + start_date + end_date
    session = str(session)
    if export_locks.get(session):
        return JSONResponse({"error": "Export already in progress."}, status_code=429)
    export_locks[session] = True
    export_progress[session] = {"status": "running", "current_day": 0, "total_days": 0, "captured": 0, "message": ""}
    def do_export():
        try:
            print(f"[Export] Starting export for r/{subreddit} from {start_date} to {end_date}")
            all_rows = []
            reddit = praw.Reddit(
                client_id=os.environ.get('REDDIT_CLIENT_ID'),
                client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
                user_agent=os.environ.get('REDDIT_USER_AGENT', 'reddit-exporter')
            )
            start_ts = int(start_dt.timestamp())
            end_ts = int((end_dt + timedelta(days=1)).timestamp())
            captured = 0
            # Fetch posts using PRAW
            praw_subreddit = reddit.subreddit(subreddit)
            for post in praw_subreddit.new(limit=limit):
                if post.created_utc < start_ts:
                    break  # Posts are in descending order; stop if before range
                if post.created_utc > end_ts:
                    continue  # Skip posts after end date
                all_rows.append([
                    post.id, 'post', str(post.author), int(post.created_utc),
                    post.title, post.selftext, post.score, post.permalink
                ])
                captured += 1
                export_progress[session]["captured"] = captured
                # Fetch comments for each post
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if getattr(comment, 'body', None) is None:
                        continue
                    all_rows.append([
                        comment.id, 'comment', str(comment.author), int(comment.created_utc),
                        '', comment.body, comment.score, comment.permalink
                    ])
                    captured += 1
                    export_progress[session]["captured"] = captured
                    if limit and captured >= limit:
                        break
                if limit and captured >= limit:
                    break
            print(f"[Export] Finished. Total captured: {captured}")
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["id", "type", "author", "created_utc", "title", "body", "score", "permalink"])
            writer.writerows(all_rows)
            export_results[session] = output.getvalue()
            export_progress[session]["status"] = "done"
            if captured == 0:
                export_progress[session]["message"] = f"No posts or comments found for r/{subreddit} from {start_date} to {end_date}."
            else:
                export_progress[session]["message"] = f"Export complete. {captured} items captured."
        except Exception as e:
            msg = f"Fatal error: {e}"
            print(f"[Export] {msg}")
            export_progress[session]["status"] = "error"
            export_progress[session]["message"] = msg
        finally:
            export_locks.pop(session, None)
    threading.Thread(target=do_export, daemon=True).start()
    return {"message": "Export started.", "session": session}

@app.get("/export_progress")
def get_export_progress(session: str):
    return export_progress.get(session, {"status": "not_started"})

@app.get("/download_export")
def download_export(session: str = ""):
    if not session:
        return JSONResponse({"error": "Missing session parameter."}, status_code=400)
    csv_data = export_results.get(session)
    if not csv_data:
        return JSONResponse({"error": "No export found for this session."}, status_code=404)
    return StreamingResponse(io.StringIO(csv_data), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=reddit_export_{session}.csv"}) 