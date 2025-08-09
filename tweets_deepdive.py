import tweepy
from sentence_transformers import SentenceTransformer, util
import os
import requests
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
from collections import namedtuple
from memory import store_session
import time
import memory_base

memory_base.load_memory()

memory = memory_base.memory

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"
TWITTER_DEV_TOKEN = os.getenv("TWITTER_DEV_TOKEN")

SUPABASE_URL= os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Twitter API
client = tweepy.Client(bearer_token=TWITTER_DEV_TOKEN)

# Load lightweight sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"TWITTER_DEV_TOKEN: {os.getenv('TWITTER_DEV_TOKEN')}")


MockTweet = namedtuple("MockTweet", ["id", "text"])

# Your skillset embedding (example)
your_skills = "AI development, embedded systems, C programming, neural networks"
your_skills_emb = model.encode(your_skills, convert_to_tensor=True)

def fetch_tweets(query, max_results=10):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=max_results)
        return tweets.data or []
    except tweepy.errors.Unauthorized as e:
        print("Twitter API Unauthorized (401):", e.response.status_code)
        print("Details:", e.response.text)
    except tweepy.TweepyException as e:
        print("Tweepy error:", str(e))
        
def fetch_tweets_mock(query=None, max_results=10):
    # Return a fixed list of mocked tweets for testing
    return [
        MockTweet(id=1, text="Most companies have an AI strategy. But leaders aren't ready. We’ve just launched the Level 5 AI Leaders Apprenticeship to close that gap: https://t.co/QEREAzMqKj Proud to help shape the next generation of AI leaders 🎉 #AILeaders #CambridgeSpark https://t.co/Z1zp10KKNj"),
        MockTweet(id=2, text="What's the weirdest thing you've used an AI writing tool for? I once had it help me craft an apology text to my houseplant after forgetting to water it for two weeks. The AI was surprisingly empathetic about my botanical neglect. What's your most unexpectedly creative use case?"),
        MockTweet(id=3, text="RT @DaddyThunder_1: @milesdeutscher Looking for a coin that shows immense strength in pullback? $BUILD is your guy. @BuildAI_erc has built…"),
    ]

 
def analyze_relevance(tweet_text):
    emb = model.encode(tweet_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(your_skills_emb, emb).item()
    return score

def generate_report(tweet_text, tweet_id):
    try:
        print(f"[REPORTING] Generating report for tweet ID {tweet_id}")
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = f"Summarize why the following tweet is a potential job lead for an AI embedded systems expert:\n\n{tweet_text}\n\nSummary:"

        data = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are an assistant that identifies job opportunities for AI embedded systems engineers."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }

        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        print(f"[REPORTING] Together AI status: {response.status_code}")

        if not response.ok:
            print("TogetherAI error:", response.status_code, response.text)
            return "[Error generating summary]"

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        print(f"[REPORTING] Report content: {content[:100]}...")  # preview first 100 chars
        return content

    except Exception as e:
        print(f"[ERROR] Report generation failed for tweet ID {tweet_id}: {e}")
        return "Report generation failed."

def save_tweet_to_supabase(tweet_id, text, score, report):
    data = {
        "id": int(tweet_id),
        "tweet_text": text,
        "score": score,
        "report": report,
    }
    response = supabase.table("job_leads").upsert(data).execute()
    if response.status_code != 200:
        print(f"Failed to save tweet {tweet_id}: {response.data}")


def tweets_deepdive_main_loop():
    query = '(help OR hiring OR "looking for") (AI OR "embedded systems" OR developer)'
    tweets = fetch_tweets(query)
    print(f"[DEBUG] Retrieved {len(tweets)} tweets")

    relevant_reports = []
    relevant_tweet_ids = []

    for tweet in tweets:
        try:
            print(f"[DEBUG] Analyzing tweet ID {tweet.id}")
            score = analyze_relevance(tweet.text)
            print(f"[DEBUG] Relevance score: {score:.2f}")

            if score > 0.5:
                print(f"[DEBUG] Relevant! Generating report for tweet ID {tweet.id}")
                report = generate_report(tweet.text, tweet.id)
                print(f"[DEBUG] Generated report: {report[:80]}...")

                save_tweet_to_supabase(tweet.id, tweet.text, score, report)
                memory_base.add_memory(text=('Tweet text: ' + tweet.text + ' Tweet report: ' + str(report) + ' Tweet score: ' + str(score)), tags=['tweet deepdive'])

                relevant_reports.append(report)
                relevant_tweet_ids.append(tweet.id)
            else:
                print(f"[DEBUG] Irrelevant. Skipping tweet ID {tweet.id}")

        except Exception as e:
            print(f"[ERROR] Error processing tweet ID {tweet.id}: {e}")

    # Now synthesize a session summary (simplified here as concatenation)
    session_summary = "\n\n".join(relevant_reports) if relevant_reports else "No relevant leads found."

    # Store the session into Chroma memory
    store_session(
        session_id="session_" + str(int(time.time())),
        query=query,
        session_summary=session_summary,
        notes="Automated job lead deepdive session",
        top_tweet_ids=relevant_tweet_ids
    )




if __name__ == "__main__":
    tweets_deepdive_main_loop()
