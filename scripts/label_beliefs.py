import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

SYSTEM_PROMPT = (
	"You are labeling belief rigidity in depressive Reddit text. "
	"Return ONLY valid JSON with keys: belief (int 1-5), discard (int 0/1). "
	"Belief scale: 1=highly rigid negative beliefs (absolute, fixed, resistant to change); "
	"5=low rigidity/high flexibility (open to uncertainty, change, alternative interpretations). "
	"Judge the writer's expressed belief structure, not emotional intensity. "
	"discard=1 if text is not depressive, irrelevant, incoherent, or too ambiguous to label; otherwise 0."
	"If discard=1, belief should be -1 since it will be ignored. Always return both keys."
)


def build_user_prompt(subreddit, text, row_id):
	return (
		f"Subreddit: {subreddit}\n"
		f"ID: {row_id}\n"
		"Text:\n"
		f"{text}\n\n"
		"Respond with JSON only. Example: {\"belief\": 3, \"discard\": 0}"
	)


def parse_json_content(content):
	content = content.strip()
	if content.startswith("```"):
		content = content.strip("`")
	start = content.find("{")
	end = content.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("No JSON object found in model output")
	payload = content[start : end + 1]
	return json.loads(payload)


def label_text(client, model, subreddit, text, row_id, max_retries=3, sleep_s=1.0):
	for attempt in range(max_retries):
		try:
			response = client.chat.completions.create(
				model=model,
				temperature=0,
				messages=[
					{"role": "system", "content": SYSTEM_PROMPT},
					{
						"role": "user",
						"content": build_user_prompt(subreddit, text, row_id),
					},
				],
			)
			content = response.choices[0].message.content
			data = parse_json_content(content)
			belief = int(data.get("belief"))
			discard = int(data.get("discard"))
			if (belief < 1 or belief > 5) and belief != -1:
				print(f"Warning: belief out of range for ID {row_id}, got {belief}. Setting discard=1.")
				raise ValueError("belief must be 1-5")
			if discard not in (0, 1):
				raise ValueError("discard must be 0 or 1")
			return belief, discard
		except Exception:
			if attempt == max_retries - 1:
				raise
			time.sleep(sleep_s * (2**attempt))


def main():
	parser = argparse.ArgumentParser(
		description="Label belief rigidity and discard flags for Dreaddit samples."
	)
	parser.add_argument(
		"--input",
		default="data/phase3_dreaddit_1000.csv",
		help="Input CSV path",
	)
	parser.add_argument(
		"--output",
		default="output/phase3_dreaddit_1000_labeled.csv",
		help="Output CSV path",
	)
	parser.add_argument(
		"--model",
		default="gpt-4o-2024-08-06",
		help="Model name for labeling",
	)
	parser.add_argument(
		"--save-every",
		type=int,
		default=25,
		help="Save progress every N rows",
	)
	args = parser.parse_args()

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise SystemExit("OPENAI_API_KEY is not set")

	df = pd.read_csv(args.input)
	required_cols = {"subreddit", "text", "id"}
	missing = required_cols - set(df.columns)
	if missing:
		raise SystemExit(f"Missing required columns: {sorted(missing)}")

	if "belief" not in df.columns:
		df["belief"] = pd.NA
	if "discard" not in df.columns:
		df["discard"] = pd.NA

	client = OpenAI(api_key=api_key)
	total = len(df)
	updated = 0

	for idx in tqdm(df.index, desc="Labeling", total=total):
		row = df.loc[idx]
		if pd.notna(row.get("belief")) and pd.notna(row.get("discard")):
			continue
		text = str(row.get("text", "")).strip()
		if not text:
			df.at[idx, "belief"] = pd.NA
			df.at[idx, "discard"] = 1
			updated += 1
			continue
		belief, discard = label_text(
			client,
			args.model,
			row.get("subreddit", ""),
			text,
			row.get("id", ""),
		)
		df.at[idx, "belief"] = belief if discard == 0 else pd.NA
		df.at[idx, "discard"] = discard
		updated += 1
		if updated % args.save_every == 0:
			df.to_csv(args.output, index=False)
			print(f"Saved {updated}/{total} labeled rows")

	df.to_csv(args.output, index=False)
	print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
	main()
