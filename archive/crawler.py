import os
import sqlite3
from dotenv import load_dotenv
import praw

DB_PATH = 'posts.db'
TABLE_NAME = 'posts'

# Fields to store
FIELDS = [
    'post_id', 'title', 'selftext', 'subreddit', 'upvotes', 'url', 'created_utc'
]

def create_table(conn):
    with conn:
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                post_id TEXT PRIMARY KEY,
                title TEXT,
                selftext TEXT,
                subreddit TEXT,
                upvotes INTEGER,
                url TEXT,
                created_utc REAL
            )
        ''')

def save_post(conn, post):
    with conn:
        conn.execute(f'''
            INSERT OR IGNORE INTO {TABLE_NAME} (post_id, title, selftext, subreddit, upvotes, url, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            post.id,
            post.title,
            post.selftext,
            post.subreddit.display_name,
            post.score,
            post.url,
            post.created_utc
        ))

def main():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    subreddit = reddit.subreddit('AskReddit')
    query = 'biggest struggle'
    for post in subreddit.search(query, sort='new', limit=20):
        save_post(conn, post)
        print(f"Saved: {post.title}")
    conn.close()

if __name__ == '__main__':
    main() 