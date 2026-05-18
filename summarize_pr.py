import sys
import re
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"
TIMEOUT = 60
MAX_CHUNK_CHARS = 2500

url = sys.argv[1]

# Validate it's a GitHub PR URL
m = re.match(r'^https://github\.com/([^/]+/[^/]+)/pull/(\d+)$', url)
if not m:
    print("Error: must be a GitHub PR URL like https://github.com/owner/repo/pull/123")
    sys.exit(1)

# Fetch the .diff
diff_url = f"https://github.com/{m.group(1)}/pull/{m.group(2)}.diff"
print(f"Fetching diff from {diff_url}")
resp = requests.get(diff_url, timeout=15)
if resp.status_code != 200:
    print(f"Error: HTTP {resp.status_code} fetching diff")
    sys.exit(1)

diff = resp.text
if not diff.strip():
    print("No changes in this PR.")
    sys.exit(0)

# Split diff into individual file chunks
chunks = re.split(r'^(?=diff --git)', diff, flags=re.MULTILINE)
chunks = [c.strip() for c in chunks if c.strip()]
print(f"Found {len(chunks)} file(s) changed\n---")

def summarize_chunk(chunk, index, total):
    # Extract filename for display
    fname_match = re.search(r'diff --git a/(\S+) b/\S+', chunk)
    fname = fname_match.group(1) if fname_match else f"file {index}"

    # Truncate chunk to MAX_CHUNK_CHARS
    body = chunk[:MAX_CHUNK_CHARS]

    prompt = f"""Summarize the key changes in this diff. Be specific about what was added, removed, or modified and why. Keep it concise but technical.

{body}"""

    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json(), fname

def clean_output(text):
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return "\n".join(lines)
    return "Could not produce a summary."

all_data = []
overall_start = requests.post(
    OLLAMA_URL,
    json={"model": MODEL, "prompt": "token", "stream": False},
    timeout=TIMEOUT
)

for i, chunk in enumerate(chunks, 1):
    print(f"[{i}/{len(chunks)}] Processing...", end=" ", flush=True)
    data, fname = summarize_chunk(chunk, i, len(chunks))
    all_data.append((data, fname))
    print(f"Done ({data.get('eval_count', '?')} tokens)")

print("\n======= Summary =======")
for idx, (data, fname) in enumerate(all_data, 1):
    print(clean_output(data["response"]))

print("\n======= Stats =======")
total_prompt_tokens = 0
total_eval_tokens = 0
total_time_s = 0
for data, fname in all_data:
    total_prompt_tokens += data.get("prompt_eval_count", 0)
    total_eval_tokens += data.get("eval_count", 0)
    total_time_s += data.get("total_duration", 0) / 1e9
tps = total_eval_tokens / total_time_s if total_time_s > 0 else 0
print(f"Prompt tokens:  {total_prompt_tokens}")
print(f"Generated:      {total_eval_tokens} tokens ({total_time_s:.2f}s, {tps:.1f} tok/s)")
