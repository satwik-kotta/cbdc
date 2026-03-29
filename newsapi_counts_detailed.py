import ast
import re
from datetime import date, timedelta
from pathlib import Path

from newsapi import NewsApiClient

src = Path("automation.py").read_text(encoding="utf-8")
api_key = re.search(r'NewsApiClient\(api_key="([^"]+)"\)', src).group(1)

module = ast.parse(src)
countries = None
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "COUNTRIES":
                countries = ast.literal_eval(node.value)
                break
    if countries is not None:
        break

newsapi = NewsApiClient(api_key=api_key)
from_date = (date.today() - timedelta(days=29)).isoformat()
to_date = date.today().isoformat()

print(f"DATE_WINDOW={from_date}..{to_date}")

for country, queries in countries.items():
    print(f"\n[{country}]")
    for q in queries:
        try:
            response = newsapi.get_everything(
                q=q,
                language="en",
                page_size=1,
                page=1,
                from_param=from_date,
                to=to_date,
            )
            status = response.get("status")
            if status != "ok":
                print(f"  QUERY='{q}' -> status={status} message={response.get('message')}")
            else:
                print(f"  QUERY='{q}' -> ok totalResults={response.get('totalResults')}")
        except Exception as e:
            print(f"  QUERY='{q}' -> EXCEPTION: {type(e).__name__}: {e}")
