import asyncio
import json
import csv
import sys
from pathlib import Path

import httpx

INPUT_CSV  = Path("target/rnc_test.csv")
OUTPUT_CSV = Path("target/rnc_test.csv")
BASE_URL   = "http://127.0.0.1:8080"
CONCURRENCY = 5  # параллельных запросов одновременно


async def analyze(client: httpx.AsyncClient, sem: asyncio.Semaphore, row: dict, idx: int, total: int) -> dict:
    async with sem:
        try:
            payload = json.loads(row["request"])
            resp = await client.post(
                f"{BASE_URL}/analyze",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"[{idx}/{total}] itemId={payload.get('itemId')} OK  shouldSplit={result.get('shouldSplit')}", flush=True)
            return result
        except Exception as e:
            print(f"[{idx}/{total}] ERROR: {e}", flush=True)
            return {"error": str(e)}


async def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"Всего запросов: {total}")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [
            analyze(client, sem, row, i + 1, total)
            for i, row in enumerate(rows)
        ]
        results = await asyncio.gather(*tasks)

    for row, result in zip(rows, results):
        row["response"] = json.dumps(result, ensure_ascii=False)

    fieldnames = ["request", "response"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово. Результаты сохранены в {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
