import os
import sqlite3
import requests
import requests.auth
import re
import time
from datetime import datetime
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from threading import Lock

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

# Global scan progress tracker
scan_progress = {}
scan_progress_lock = Lock()

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
                category TEXT,
                upvotes INTEGER,
                created_at TEXT,
                comment_url TEXT
            )
        ''')

# Helper: Save a matched problem to DB (deduplication via UNIQUE)
def save_problem(conn, table, data):
    with conn:
        conn.execute(f'''
            INSERT OR IGNORE INTO {table} (subreddit, post_title, post_id, comment_id, author, problem, category, upvotes, created_at, comment_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['subreddit'],
            data['post_title'],
            data['post_id'],
            data['comment_id'],
            data['author'],
            data['problem'],
            data['category'],
            data['upvotes'],
            data['created_at'],
            data['comment_url']
        ))

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

def extract_and_categorize_problem_openai(text, categories=None):
    """
    Uses OpenAI to extract a quoted problem statement and assign a category in a single call.
    If categories is None, only extracts the problem.
    Returns (problem, category) or (None, None) if no problem is found.
    """
    if not openai or not OPENAI_API_KEY or not text.strip():
        return None, None
    openai.api_key = OPENAI_API_KEY
    if categories:
        cat_list = ', '.join(categories)
        prompt = (
            "Does the following text describe a real problem, pain point, or challenge? "
            f"If yes, quote the sentence or two (exactly as written) that best describe the problem, and assign it to one of these categories: {cat_list}. "
            "Reply in this format:\nPROBLEM: \"quoted problem\"\nCATEGORY: category_name\nIf not, reply 'NO PROBLEM'.\n\n"
            f"Text: '''{text}'''"
        )
    else:
        prompt = (
            "Does the following text describe a real problem, pain point, or challenge? "
            "If yes, quote the sentence or two (exactly as written) that best describe the problem. "
            "If not, reply 'NO PROBLEM'.\n\n"
            f"Text: '''{text}'''"
        )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at identifying and categorizing real problems in online discussions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0
        )
        result = response.choices[0].message.content.strip()  # type: ignore
        if result.upper() == 'NO PROBLEM' or not result:
            return None, None
        if categories:
            # Expecting format: PROBLEM: "..."\nCATEGORY: ...
            lines = result.split('\n')
            problem = None
            category = None
            for line in lines:
                if line.strip().startswith('PROBLEM:'):
                    problem = line.split('PROBLEM:',1)[1].strip().strip('"')
                elif line.strip().startswith('CATEGORY:'):
                    category = line.split('CATEGORY:',1)[1].strip()
            if not problem:
                # fallback: try to find quoted text
                import re
                m = re.search(r'"([^"]+)"', result)
                if m:
                    problem = m.group(1)
            return problem, category
        else:
            # Only problem extraction
            import re
            m = re.search(r'"([^"]+)"', result)
            if m:
                return m.group(1), None
            return result, None
    except Exception as e:
        print(f"OpenAI API error (extract & categorize): {e}")
        return None, None

@app.get("/scan_progress")
def get_scan_progress(subreddit: str):
    key = subreddit.lower()
    with scan_progress_lock:
        prog = scan_progress.get(key, None)
    if prog is None:
        return {"status": "idle"}
    return prog

@app.post("/scan")
def scan_subreddit(req: ScanRequest):
    subreddit = req.subreddit.strip()
    batch_size = req.batch_size
    total_posts = req.total_posts
    table = f"pain_points_{subreddit.lower()}"
    token = get_token()
    headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}
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
                # --- Scan the original post body ---
                post_body = post_data.get('selftext', '')
                if post_body.strip():
                    if not categories and len(sample_problems) < sample_limit:
                        problem, _ = extract_and_categorize_problem_openai(post_body)
                        if problem:
                            sample_problems.append(problem)
                            if len(sample_problems) >= sample_limit:
                                categories = get_or_generate_categories(subreddit, sample_problems)
                    else:
                        problem, category = extract_and_categorize_problem_openai(post_body, categories)
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
                    if not body.strip():
                        continue
                    if not categories and len(sample_problems) < sample_limit:
                        problem, _ = extract_and_categorize_problem_openai(body)
                        if problem:
                            sample_problems.append(problem)
                            if len(sample_problems) >= sample_limit:
                                categories = get_or_generate_categories(subreddit, sample_problems)
                    else:
                        problem, category = extract_and_categorize_problem_openai(body, categories)
                        if problem and category:
                            n_problems += 1
                            with scan_progress_lock:
                                scan_progress[key]["problems_found"] = max(scan_progress[key]["problems_found"], n_problems)
                            data = {
                                'subreddit': subreddit,
                                'post_title': post_title,
                                'post_id': post_id,
                                'comment_id': c.get('id'),
                                'author': c.get('author'),
                                'problem': problem,
                                'category': category,
                                'upvotes': c.get('score', 0),
                                'created_at': datetime.utcfromtimestamp(c.get('created_utc', 0)).isoformat() if c.get('created_utc') else None,
                                'comment_url': make_comment_url(subreddit, post_id, c.get('id'))
                            }
                            save_problem(conn, table, data)
                time.sleep(1)
            if not after:
                break
        # If categories were never generated (not enough samples), generate with what we have
        if not categories:
            categories = get_or_generate_categories(subreddit, sample_problems)
    finally:
        with scan_progress_lock:
            scan_progress[key]["status"] = "done"
        conn.close()
    return {"message": f"✅ Scan complete: {n_problems} problems added from {n_posts} posts."}

@app.get("/summary")
def get_summary(subreddit: str = Query(None)):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    all_rows = []
    if subreddit:
        # Only aggregate from the table for the selected subreddit
        table_name = f"pain_points_{subreddit}"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            cursor.execute(f"SELECT category, problem, upvotes, author, comment_url, subreddit FROM {table_name}")
            all_rows.extend(cursor.fetchall())
    else:
        # No subreddit selected, return empty list
        conn.close()
        return []
    summary = {}
    for cat, problem, upvotes, author, url, subreddit in all_rows:
        if not cat:
            cat = 'Uncategorized'
        if cat not in summary:
            summary[cat] = {"category": cat, "total_upvotes": 0, "problems": []}
        summary[cat]["total_upvotes"] += upvotes
        summary[cat]["problems"].append({
            "problem": problem,
            "upvotes": upvotes,
            "author": author,
            "comment_url": url,
            "subreddit": subreddit
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
def recategorize_subreddit(subreddit: str):
    table = f"pain_points_{subreddit.lower()}"
    cat_table = f"categories_{subreddit.lower()}"
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()
    # Load categories
    cursor.execute(f"SELECT category FROM {cat_table}")
    categories = [row[0] for row in cursor.fetchall()]
    if not categories:
        conn.close()
        return {"message": f"No categories found for subreddit {subreddit}."}
    # Get all problems
    cursor.execute(f"SELECT id, problem FROM {table}")
    rows = cursor.fetchall()
    updated = 0
    for row_id, problem in rows:
        new_cat = assign_category_openai(problem, categories, subreddit)
        cursor.execute(f"UPDATE {table} SET category=? WHERE id=?", (new_cat, row_id))
        updated += 1
    conn.commit()
    conn.close()
    return {"message": f"Recategorized {updated} problems in {subreddit} using 10-category set."}

# Serve index.html at the root path
@app.get("/")
def read_index():
    return FileResponse("static/index.html") 