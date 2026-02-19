import argparse, json, sys, shutil, os
from typing import Literal, Dict, Any, List, Optional
import requests
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import polars as pl

# load environment variables, such as API keys
load_dotenv()

############################
# Important info
########## Time r1 ##########
# Direction: Claims - Posts & Time: 28 Days before claim up to date of claim
########## Time r2 ##########
# Direction: Claim - Posts & Time: Date of Claim and 28 days after
############################


#############################
# Prompts and prompt logic
#############################

# defining the pair class
class Pair(BaseModel):
    id: str
    claim: str
    tweet: str
    # putting the two orders as literals to force certain input
    order: Literal["tweet-claim", "claim-tweet"]
    # and the gold labels as well
    # None has to be deleted as option for gold_label once the gold_label is defined by Prolific users' results
    gold_label: Optional[Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION", None]] = None


# defining the Prediction class, a schema for the LLM's prediction output in JSON, validating parsed label
class Prediction(BaseModel):
    label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


# Prompts

# System Prompt

# Claim -> Tweet (other direction doesn't exist)
SYSTEM_C2T = """
Which of the following best describes the relationship between TWEET and CLAIM?
You must provide a final choice as ENTAILMENT, NEUTRAL, or CONTRADICTION.

If TWEET is true:
(ENTAILMENT) then CLAIM is also true.
(NEUTRAL) CLAIM cannot be said to be true or false.
(CONTRADICTION) then CLAIM is false.

Important:
- Output valid JSON only: {"label": "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION"}.
"""

### User Prompts
## Minimal prompt, annotation-only (zero-shot)
# Claim -> Tweet
USER_MIN_C2T = """Classify the relation between the following CLAIM and TWEET.\n\nCLAIM:\n{claim}\n\nTWEET:\n{tweet}\n\nReturn only one JSON object: {{\"label\":\"ENTAILMENT\"}} or {{\"label\":\"NEUTRAL\"}} or {{\"label\":\"CONTRADICTION\"}}."""


### Building Prompts based on order and desired user prompts format

# System prompt based on order (T->C / C->T)
def build_system_prompt(order: str) -> str:
    return SYSTEM_C2T

# Alternative for explanatory answers:
def build_user_prompt_min(pair: Pair) -> str:
    return USER_MIN_C2T.format(tweet=pair.tweet, claim=pair.claim)
    '''# Choose the correct base template (since we only have C->T, keeping only USER_MIN_C2T)
    base = USER_MIN_C2T
    # If we want an explanation, swap the JSON-instruction line for a version that includes "rationale"

    # IMPORTANT: do not use .format because the prompt contains JSON braces.
    # Replace only our explicit placeholders for tweet and claim.
    base = base.replace("{tweet}", pair.tweet).replace("{claim}", pair.claim)
    return base'''


#############################
# LLM logic
#############################


### Calls to the LLMs

# OpenAI
# using parameters as they were modified in Choi, Ferrara: Automated Claim Matching with Large Language Models:
# Empowering Fact-Checkers in the Fight Against Misinformation
# e.g. base_url: hosting the LLM locally with local host as default
# temperature 0.0 as default
# here: Call to OpenAI only for zero-shot prompt w/o explanation!!!
def call_openai(base_url: str, api_key: str, model: str, system: str, user: str, temperature: float) -> str:
    # v1-> version 1 of public OpenAI API; chat/completions -> routing it to the generation of chat completions
    url = base_url.rstrip("/") + "/v1/chat/completions"
    # Resolve API key: if user passed "ollama", take the real key from OLLAMA_API_KEY env var (recommended for Ollama Cloud via local /v1 proxy)
    eff_key = os.environ.get("OLLAMA_API_KEY") if api_key == "ollama" else api_key
    # Authorization via own bearer API key; content type defined as json, such that server knows how to parse it (per
    # https request)
    headers = {"Authorization": f"Bearer {eff_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        # Keeping output short to 16 tokens -> no explanations output
        "max_tokens": 16,
        '''for explanation output, you may do this:
        max_out = 16 if not explain else 256
        payload["max_tokens"] = max_out'''

        "response_format": {"type": "json_object"}
    }
    # wait for up to five minutes to process request at max
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    # throws exception if HTTP response status is erroneous
    r.raise_for_status()
    # Convert the HTTP response body from JSON
    data = r.json()
    # categorizes output; trim leading/trailing whitespace
    return data["choices"][0]["message"]["content"].strip()


# Ollama - old version that did it locally but could also send it to cloud model
'''
def call_ollama(model: str, system: str, user: str, temperature: float) -> str:
    # Ollamas default port (11434) on Local host, opening chat endpoint
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature},
        # wait until entire response is generated, then send it
        "stream": False
    }
    # longer timeout for ollama bcs local models (Ollama) often need more time to respond than hosted APIs
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
'''


# Ollama - as on website
def call_ollama(model: str, system: str, user: str, temperature: float) -> str:
    url = "https://ollama.com/api/chat"
    headers = {'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature},
        "stream": False
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


### Parsing & evaluation
def parse_label(raw: str) -> Prediction:
    raw = raw.strip()
    # just in case label is hidden in output -> find label {LABEL}
    if not raw.startswith("{"):
        start = raw.find("{");
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    obj = json.loads(raw)
    # make a Prediction object from the dict (keys/vals passed in with **obj)
    return Prediction(**obj)


def eval_metrics(rows: List[Dict[str, Any]]):
    golds = [r["gold_label"] for r in rows if r.get("gold_label") is not None]
    preds = [r["prediction"] for r in rows if r.get("gold_label") is not None]
    if not golds:
        print("No gold labels provided; skipping evaluation.")
        return
    labels = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    # define label order & index map
    idx = {lab: i for i, lab in enumerate(labels)}
    # create empty fusion matrix
    cm = [[0] * 3 for _ in range(3)]
    correct = 0
    # fill in confusion matrix
    for g, p in zip(golds, preds):
        if g == p: correct += 1
        cm[idx[g]][idx[p]] += 1
    # calculating accuracy
    acc = correct / len(golds)
    # accuracy rounded to 3 decimals
    print(f"\nAccuracy: {acc:.3f}")
    print("Confusion matrix (rows = gold, cols = pred):")
    print("            ENT   NEU   CON")
    # prints confusion matrix line by line
    for i, lab in enumerate(labels):
        print(f"{lab[:3]:<3}   {cm[i][0]:>5} {cm[i][1]:>5} {cm[i][2]:>5}")


#############################
# Runner / CLI
#############################

def pairs_from_polars_df(df: pl.DataFrame, assume_order: str = "claim-tweet", start_index: int = 0) -> List[Pair]:
    pairs: List[Pair] = []
    for idx, row in enumerate(df.iter_rows(named=True)):
        pairs.append(
            Pair(
                id=str(start_index + idx),
                claim=row["claim_text"],
                tweet=row["post_text"],
                order=assume_order,
                gold_label=None
            )
        )
    return pairs



# Streaming/batched dataset processor: read, predict, and write in chunks
def process_dataset_streaming(csv_path: str, name: str, args) -> None:
    out_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    stem, ext = os.path.splitext(base_name)

    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    annotations_csv_path = os.path.join(out_dir, f"predictions/{stem}_predictions{ext}")
    jsonl_path = os.path.join(out_dir, f"predictions/predictions_{name}.jsonl")

    # If files exist from a previous run, start fresh
    if os.path.exists(annotations_csv_path):
        os.remove(annotations_csv_path)
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    # Discover column names from header only once
    header_df = pl.read_csv(csv_path, infer_schema_length=0, n_rows=0)
    columns = header_df.columns
    if "claim_text" not in columns or "post_text" not in columns:
        raise ValueError(f"CSV {csv_path} must contain columns 'claim_text' and 'post_text'.")

    total_rows_processed = 0
    batch_size = int(args.batch_size)
    max_rows = args.max_rows if args.max_rows is not None else float("inf")
    first_chunk = True

    with open(jsonl_path, "w", encoding="utf-8") as fout_jsonl:
        while total_rows_processed < max_rows:
            # For first chunk use header, for subsequent chunks skip header (+1) and set has_header=False
            if total_rows_processed == 0:
                df_chunk = pl.read_csv(
                    csv_path,
                    infer_schema_length=0,
                    n_rows=min(batch_size, int(max_rows)),
                    has_header=True,
                )
            else:
                df_chunk = pl.read_csv(
                    csv_path,
                    infer_schema_length=0,
                    n_rows=min(batch_size, int(max_rows) - total_rows_processed),
                    skip_rows=total_rows_processed + 1,
                    has_header=False,
                    new_columns=columns,
                )

            if df_chunk.is_empty():
                break

            # Build Pair objects with global indices
            pairs = pairs_from_polars_df(df_chunk, assume_order="claim-tweet", start_index=total_rows_processed)

            # Prepare predictions list aligned to this chunk only
            chunk_predictions: List[Optional[str]] = [None] * df_chunk.height
            rows_out: List[Dict[str, Any]] = []

            for pair in pairs:
                system_prompt = build_system_prompt(pair.order)
                user_prompt = build_user_prompt_min(pair)
                print(pair.id)

                try:
                    if args.provider == "openai":
                        content = call_openai(args.base_url, args.api_key, args.model, system_prompt, user_prompt, args.temperature)
                    else:
                        content = call_ollama(args.model, system_prompt, user_prompt, args.temperature)
                    pred = parse_label(content)
                except (requests.HTTPError, requests.ConnectionError) as e:
                    print(f"[{pair.id}] HTTP Error: {e}", file=sys.stderr)
                    continue
                except (json.JSONDecodeError, ValidationError) as e:
                    try:
                        retry_user = user_prompt + '\n\nIMPORTANT: Respond with exactly this JSON schema: {"label":"ENTAILMENT"|"NEUTRAL"|"CONTRADICTION"}.'
                        if args.provider == "openai":
                            content = call_openai(args.base_url, args.api_key, args.model, system_prompt, retry_user, args.temperature)
                        else:
                            content = call_ollama(args.model, system_prompt, retry_user, args.temperature)
                        pred = parse_label(content)
                    except Exception as e2:
                        print(f"[{pair.id}] Parse error after retry: {e2}", file=sys.stderr)
                        continue

                # Map to chunk-local index
                try:
                    global_idx = int(pair.id)
                    print(global_idx)
                except ValueError:
                    global_idx = None

                if global_idx is not None:

                    local_idx = global_idx - total_rows_processed
                    print(local_idx)
                    if 0 <= local_idx < len(chunk_predictions):
                        chunk_predictions[local_idx] = pred.label
                        print(chunk_predictions[local_idx])
                        print("L2l")

                record = {
                    "id": pair.id,
                    "order": pair.order,
                    "claim": pair.claim,
                    "tweet": pair.tweet,
                    "prediction": pred.label,
                    "gold_label": pair.gold_label,
                }
                print(f"Record", record)
                fout_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_out.append(record)
                print(rows_out)
                print(f"[{pair.id}] {pred.label}")

            # Attach predictions to this chunk and append to CSV
            annotated_chunk = df_chunk.with_columns(pl.Series("prediction", chunk_predictions))
            # Append-friendly writing
            with open(annotations_csv_path, "a", encoding="utf-8", newline="") as fout_csv:
                annotated_chunk.write_csv(fout_csv, include_header=first_chunk)
            first_chunk = False

            total_rows_processed += df_chunk.height
            if df_chunk.height < batch_size:
                break

    print(f"Finished: {name}: wrote {jsonl_path}")
    print(f"Wrote annotations CSV (streamed): {annotations_csv_path}")


def main():
    # adding requirement arguments with options (and help)
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True,
                        help="Model name, e.g., qwen:7b (ollama) or gpt-oss-12b (openai-compatible)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Determine how deterministic the model is")
    parser.add_argument("--base_url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible base URL (use http://localhost:11434/v1 for Ollama local proxy / cloud offload; ignored for --provider ollama")
    parser.add_argument("--api_key", default="not-needed", help="API key (ignored for ollama)")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Process the dataset in chunks of this many rows")
    parser.add_argument("--max_rows", type=int, default=None, help="(Optional) Limit total rows processed (for quick tests)")
    args = parser.parse_args()

    # CSV base path
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    ############ r1 ############
    print("Now processing matches r1")
    r1_path = os.path.join(base_path, "sbert_datasets/matches_sbert_r1.csv")
    process_dataset_streaming(r1_path, name="r1", args=args)
    print("\nFinished processing matches r1")

    ############ r2 ############
    print("Now processing matches r2")
    r2_path = os.path.join(base_path, "sbert_datasets/matches_sbert_r2.csv")
    process_dataset_streaming(r2_path, name="r2", args=args)
    print("\nFinished processing matches r2")

    print("\n Completed all matches processing.")


if __name__ == "__main__":
    main()
