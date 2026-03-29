import ast
import re
from datetime import date, timedelta
from pathlib import Path

from newsapi import NewsApiClient

SRC_PATH = Path("automation.py")


def load_api_key(source_text: str) -> str:
    match = re.search(r'NewsApiClient\(api_key="([^"]+)"\)', source_text)
    if not match:
        raise RuntimeError("API key not found in automation.py")
    return match.group(1)


def load_countries(source_text: str) -> dict:
    module = ast.parse(source_text)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "COUNTRIES":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, dict):
                        return value
    raise RuntimeError("COUNTRIES dict not found in automation.py")


def main() -> None:
    source_text = SRC_PATH.read_text(encoding="utf-8")
    api_key = load_api_key(source_text)
    countries = load_countries(source_text)

    newsapi = NewsApiClient(api_key=api_key)

    # Free plan on this key currently allows going back to 2026-02-26.
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=29)).isoformat()

    print(f"DATE_WINDOW={from_date}..{to_date}")
    print("COUNTRY|queries|extracted_raw|extracted_unique|queries_with_errors|queries_truncated")

    for country, queries in countries.items():
        raw_count = 0
        unique_urls = set()
        error_count = 0
        truncated_count = 0

        for q in queries:
            try:
                response = newsapi.get_everything(
                    q=q,
                    language="en",
                    page_size=100,
                    page=1,
                    from_param=from_date,
                    to=to_date,
                )

                if response.get("status") != "ok":
                    error_count += 1
                    continue

                articles = response.get("articles", []) or []
                raw_count += len(articles)

                for article in articles:
                    url = article.get("url")
                    if url:
                        unique_urls.add(url)

                total_results = int(response.get("totalResults") or 0)
                if total_results > len(articles):
                    truncated_count += 1

            except Exception:
                error_count += 1

        print(
            f"{country}|{len(queries)}|{raw_count}|{len(unique_urls)}|{error_count}|{truncated_count}"
        )


if __name__ == "__main__":
    main()
