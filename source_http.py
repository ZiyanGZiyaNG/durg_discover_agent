"""外部資料來源共用 HTTP helper：timeout、retry、backoff。"""
import time
from typing import Any, Dict, Optional

import requests

USER_AGENT = "ncku-drug-discovery-agent1/0.1 (academic research)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    """外部來源查詢失敗（重試後仍失敗）。"""


def _retry_after(resp: requests.Response, default: float) -> float:
    try:
        return min(float(resp.headers.get("Retry-After", default)), 30.0)
    except ValueError:
        return default


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries + 1):
        wait = backoff ** attempt
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=timeout,
            )
            if resp.status_code in RETRY_STATUS:
                last_err = f"HTTP {resp.status_code}"
                wait = _retry_after(resp, wait)
            else:
                resp.raise_for_status()
                return resp.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = repr(exc)
        except (requests.HTTPError, ValueError) as exc:
            # 4xx 或 JSON 解析失敗：重試沒有意義，直接丟出
            raise SourceError(f"{url}: {exc!r}") from exc
        if attempt < max_retries:
            time.sleep(wait)
    raise SourceError(f"{url}: 重試 {max_retries} 次後仍失敗（{last_err}）")