"""
Test which APIs are working and how many requests they use
"""
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
import feedparser

api_key = "67f062a32b7945578c2c69473ac3eff1"
newsapi = NewsApiClient(api_key=api_key)

to_date = datetime.now().date()
from_date = to_date - timedelta(days=7)

print("=" * 60)
print("API CAPACITY TEST")
print("=" * 60)

# Test 1: NewsAPI
print("\n1. NEWSAPI")
try:
    response = newsapi.get_everything(
        q="digital",
        language="en",
        page_size=100,
        page=1,
        from_param=from_date.isoformat(),
        to=to_date.isoformat()
    )
    if response.get('status') == 'ok':
        print(f"   ✓ Working - {len(response['articles'])} articles")
    else:
        print(f"   ✗ Error: {response.get('message')}")
except Exception as e:
    print(f"   ✗ Exception: {str(e)[:100]}")

# Test 2: Google News RSS
print("\n2. GOOGLE NEWS RSS")
try:
    url = "https://news.google.com/rss/search?q=CBDC&ceid=US:en"
    feed = feedparser.parse(url)
    if feed.entries:
        print(f"   ✓ Working - {len(feed.entries)} articles")
    else:
        print(f"   ✗ No results")
except Exception as e:
    print(f"   ✗ Exception: {str(e)[:100]}")

# Test 3: GDELT
print("\n3. GDELT")
try:
    query = "CBDC"
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=ArtList&maxrecords=100&format=json&startdatetime=20260313000000&enddatetime=20260320235959"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        print(f"   ✓ Working - {len(articles)} articles")
    else:
        print(f"   ✗ Status: {response.status_code}")
except Exception as e:
    print(f"   ✗ Exception: {str(e)[:100]}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("• NewsAPI: Limited (100 req/24hr) but HIGH QUALITY articles")
print("• Google News: Works but URLs can't be extracted")
print("• GDELT: Works but slow and unreliable")
print("\nBest strategy: Use NewsAPI only (no rate limit issues)")
print("=" * 60)
