from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def github_request(url: str, token: str, method: str = "GET", data: dict | None = None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "daytrade-ranker-jp",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def main() -> None:
    alert = json.loads((ROOT / "docs" / "alert.json").read_text(encoding="utf-8"))
    if not alert.get("announce"):
        print("No announcement: conditions were not met")
        return

    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    api = f"https://api.github.com/repos/{repository}"
    query = urllib.parse.quote(f'repo:{repository} is:issue in:title "{alert["title"]}"')
    existing = github_request(f"https://api.github.com/search/issues?q={query}", token)
    if any(item.get("title") == alert["title"] for item in existing.get("items", [])):
        print("Announcement already exists")
        return

    issue = github_request(
        f"{api}/issues", token, method="POST",
        data={"title": alert["title"], "body": alert["body"]},
    )
    print(f"Created {issue['html_url']}")


if __name__ == "__main__":
    main()
