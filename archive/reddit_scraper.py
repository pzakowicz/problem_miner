import os
import requests
import requests.auth
import sqlite3
import time
import re
from datetime import datetime
from dotenv import load_dotenv

# Optional: OpenAI for categorization
try:
    import openai
except ImportError:
    openai = None

# Load environment variables from .env
load_dotenv()

CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')
USERNAME = os.getenv('REDDIT_USERNAME')
PASSWORD = os.getenv('REDDIT_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

if (CLIENT_ID is None or CLIENT_SECRET is None or USER_AGENT is None or
    USERNAME is None or PASSWORD is None):
    raise ValueError('REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, and REDDIT_PASSWORD must be set in your .env file.')

CLIENT_ID = str(CLIENT_ID)
CLIENT_SECRET = str(CLIENT_SECRET)
USER_AGENT = str(USER_AGENT)
USERNAME = str(USERNAME)
PASSWORD = str(PASSWORD)

# Pain point keywords to search for in comments
PAIN_KEYWORDS = [
    'problem', 'struggle', 'issue', "can't", 'hate', 'wish', 'barrier', 'challenge'
]

DB_PATH = 'reddit_pain_points.db'
TABLE_NAME = 'pain_points'

# Reddit API endpoints
TOKEN_URL = 'https://www.reddit.com/api/v1/access_token'
BASE_URL = 'https://oauth.reddit.com'

# Helper: Get OAuth token
def get_token():
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {
        'grant_type': 'password',
        'username': USERNAME,
        'password': PASSWORD
    }
    headers = {'User-Agent': USER_AGENT}
    try:
        res = requests.post(TOKEN_URL, auth=auth, data=data, headers=headers)
        res.raise_for_status()
        token = res.json()['access_token']
        return token
    except Exception as e:
        print(f"Error obtaining token: {e}")
        exit(1)

# Helper: Create SQLite table (drops and recreates for schema change)
def create_table(conn):
    with conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subreddit TEXT,
                post_title TEXT,
                post_id TEXT,
                comment_id TEXT,
                author TEXT,
                problem TEXT,
                category TEXT,
                upvotes INTEGER,
                created_at TEXT,
                comment_url TEXT
            )
        ''')

# Helper: Save a matched problem to DB
def save_problem(conn, data):
    with conn:
        conn.execute(f'''
            INSERT INTO {TABLE_NAME} (subreddit, post_title, post_id, comment_id, author, problem, category, upvotes, created_at, comment_url)
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
        # Recursively process replies
        if c.get('replies') and isinstance(c['replies'], dict):
            extract_comments(c['replies']['data']['children'], results)

# Extract all relevant sentences containing pain keywords
def extract_problem_sentences(comment_body, keywords):
    sentences = re.split(r'(?<=[.!?]) +', comment_body)
    return [s.strip() for s in sentences if any(kw in s.lower() for kw in keywords)]

# Construct a direct Reddit comment URL
def make_comment_url(subreddit, post_id, comment_id):
    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/_/{comment_id}"

# Categorize a problem using OpenAI (if available)
def categorize_problem_openai(problem):
    if not openai or not OPENAI_API_KEY:
        return "uncategorized"
    openai.api_key = OPENAI_API_KEY
    prompt = (
        "Categorize the following fitness-related problem in one or two words "
        "(e.g., injury, consistency, progress, motivation, equipment, etc.):\n"
        f"\"{problem}\"\nCategory:"
    )
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes fitness-related problems."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8,
            temperature=0
        )
        category = response['choices'][0]['message']['content'].strip()
        # Only keep the first word or two
        return category.split('\n')[0].split(',')[0].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "uncategorized"

# Main scraping logic
def main():
    subreddit = 'HybridAthlete'  # Change as needed
    batch_size = 20              # How many posts per API call (max 100 for Reddit API, but 20 is safe)
    total_posts_to_fetch = 100   # Total number of posts you want to fetch
    sort = 'hot'  # or 'top'
    token = get_token()
    headers = {'Authorization': f'bearer {token}', 'User-Agent': USER_AGENT}

    # Connect to DB and ensure table exists (drops and recreates for schema change)
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    n_posts = 0
    n_problems = 0
    after = None

    while n_posts < total_posts_to_fetch:
        url = f"{BASE_URL}/r/{subreddit}/{sort}?limit={batch_size}"
        if after:
            url += f"&after={after}"
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()['data']
            posts = data['children']
            after = data.get('after')
            if not posts:
                break
        except Exception as e:
            print(f"Error fetching posts: {e}")
            break

        for post in posts:
            n_posts += 1
            if n_posts > total_posts_to_fetch:
                break
            post_data = post['data']
            post_id = post_data['id']
            post_title = post_data.get('title', '')
            # Fetch comments for this post
            comments_url = f"{BASE_URL}/r/{subreddit}/comments/{post_id}?limit=500"
            try:
                c_resp = requests.get(comments_url, headers=headers)
                c_resp.raise_for_status()
                comment_listing = c_resp.json()
                if len(comment_listing) < 2:
                    continue
                all_comments = []
                extract_comments(comment_listing[1]['data']['children'], all_comments)
            except Exception as e:
                print(f"Error fetching comments for post {post_id}: {e}")
                continue
            if not all_comments:
                continue
            for c in all_comments:
                body = c.get('body', '')
                problem_sentences = extract_problem_sentences(body, PAIN_KEYWORDS)
                for problem_text in problem_sentences:
                    n_problems += 1
                    category = categorize_problem_openai(problem_text)
                    data = {
                        'subreddit': subreddit,
                        'post_title': post_title,
                        'post_id': post_id,
                        'comment_id': c.get('id'),
                        'author': c.get('author'),
                        'problem': problem_text,
                        'category': category,
                        'upvotes': c.get('score', 0),
                        'created_at': datetime.utcfromtimestamp(c.get('created_utc', 0)).isoformat() if c.get('created_utc') else None,
                        'comment_url': make_comment_url(subreddit, post_id, c.get('id'))
                    }
                    save_problem(conn, data)
            # Be polite to Reddit API
            time.sleep(1)

        if not after:
            break  # No more pages

    conn.close()
    print(f"Checked {n_posts} posts, found {n_problems} pain point problems.")

if __name__ == '__main__':
    main() 