import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o-2024-08-06"

SYSTEM_PROMPT = (
	"You are a rigorous evaluator of CBT-style therapeutic responses for a research study. "
"Evaluate each response using the rubric below. Score each category independently on a 0–4 scale using the operational definitions provided. "
"Assign the score that best matches the evidence present in the text. Do not infer intentions that are not explicitly supported by the response. "
"Do not give partial credit outside the defined scale. Apply the rubric consistently across all responses.\n\n"

"Scale:\n"
"0: Absent or counterproductive. The response contradicts CBT principles, reinforces maladaptive beliefs, or provides no relevant therapeutic content.\n"
"1: Minimal presence. A vague or superficial attempt that lacks clarity, specificity, or correct CBT structure.\n"
"2: Partial implementation. Some relevant CBT elements appear but are incomplete, weakly expressed, or inconsistently applied.\n"
"3: Adequate implementation. Clear and appropriate use of the CBT element with reasonable specificity and coherence.\n"
"4: Strong implementation. Precise, well-structured, and therapeutically appropriate use of the CBT element with clear relevance to the user's statement.\n\n"

"Categories:\n"

"- therapy_quality:\n"
"Overall CBT quality of the response. Consider clarity, structure, appropriateness of tone, and whether the response follows a cognitive restructuring framework "
"(acknowledging emotion, identifying cognition, encouraging reflection, and suggesting constructive reframing or action).\n"

"- identifies_belief:\n"
"Does the response explicitly identify or restate the underlying belief, interpretation, or automatic thought expressed by the user? "
"High scores require clear recognition of the belief as a thought or interpretation rather than a fact.\n"

"- asks_evidence:\n"
"Does the response encourage examination of evidence, alternative explanations, or cognitive flexibility? "
"This may include questions prompting reflection, requests for evidence, or presentation of plausible alternative interpretations.\n"

"- avoids_validation:\n"
"Does the response avoid endorsing hopeless, absolute, or distorted beliefs as factual? "
"Appropriate empathy is allowed, but the response should not reinforce cognitive distortions or present them as accurate conclusions.\n"

"- proposes_next_step:\n"
"Does the response suggest a concrete, realistic next step consistent with CBT practice? "
"This may include a small behavioral experiment, reflective exercise, journaling prompt, or actionable coping strategy.\n"
)


def _build_batch_payload(items: List[Dict[str, str]]) -> str:
	payload = []
	for item in items:
		payload.append(
			{
				"id": item["id"],
				"text": item["text"],
				"response": item["response"],
			}
		)
	return json.dumps(payload, ensure_ascii=True)


def _extract_json_array(raw_text: str) -> str:
	text = raw_text.strip()
	if not text:
		raise ValueError("Model output is empty.")
	start = text.find("[")
	end = text.rfind("]")
	if start == -1 or end == -1 or end < start:
		raise ValueError("Model output does not contain a JSON array.")
	return text[start : end + 1]


def _parse_scores(raw_text: str) -> List[Dict[str, int]]:
	json_text = _extract_json_array(raw_text)
	data = json.loads(json_text)
	if not isinstance(data, list):
		raise ValueError("Model output is not a list.")
	return data


def score_batch(
	client: OpenAI,
	model: str,
	items: List[Dict[str, str]],
	max_retries: int,
	retry_sleep: float,
) -> List[Dict[str, int]]:
	user_payload = _build_batch_payload(items)
	for attempt in range(1, max_retries + 1):
		try:
			response = client.responses.create(
				model=model,
				input=[
					{"role": "system", "content": SYSTEM_PROMPT},
					{
						"role": "user",
						"content": (
							"Score each item. Return ONLY valid JSON (no markdown). "
							"Output must be a JSON array of objects with keys: id, "
							"therapy_quality, identifies_belief, asks_evidence, avoids_validation, "
							"proposes_next_step. The id must match the input item id.\n\n"
							f"ITEMS_JSON: {user_payload}"
						),
					},
				],
			)
			raw_text = getattr(response, "output_text", None)
			if raw_text is None:
				raise ValueError("No output_text returned from model.")
			return _parse_scores(raw_text)
		except Exception as exc:
			if attempt >= max_retries:
				raise
			time.sleep(retry_sleep * attempt)


def score_dataframe(
	df: pd.DataFrame,
	client: OpenAI,
	model: str,
	text_col: str,
	response_col: str,
	column_map: Dict[str, str],
	batch_size: int,
	max_retries: int,
	retry_sleep: float,
) -> pd.DataFrame:
	outputs: Dict[int, Dict[str, int]] = {}

	rows = df[[text_col, response_col]].to_dict("records")
	for start in tqdm(
		range(0, len(rows), batch_size),
		desc="Scoring batches",
		unit="batch",
	):
		batch = []
		for idx, row in enumerate(rows[start : start + batch_size], start=start):
			batch.append({"id": idx, "text": row[text_col], "response": row[response_col]})

		scores = score_batch(client, model, batch, max_retries, retry_sleep)
		for item in scores:
			outputs[int(item["id"])] = {
				"therapy_quality": int(item["therapy_quality"]),
				"identifies_belief": int(item["identifies_belief"]),
				"asks_evidence": int(item["asks_evidence"]),
				"avoids_validation": int(item["avoids_validation"]),
				"proposes_next_step": int(item["proposes_next_step"]),
			}

	df = df.copy()
	for key, col_name in column_map.items():
		df[col_name] = [
			outputs.get(i, {}).get(key) for i in range(len(df))
		]
	return df


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Grade steered CBT responses with an LLM judge, in batches.")
	parser.add_argument("--input-csv", help="Input CSV with text and response columns.")
	parser.add_argument("--output-csv", help="Output CSV path with score column added.")
	parser.add_argument(
		"--input-dir",
		default="data/runs/final",
		help="Directory of run CSVs (steering_layer_[layer]_alpha_[alpha].csv).",
	)
	parser.add_argument(
		"--output-dir",
		default="output/graded_runs/final",
		help="Directory to write graded CSVs.",
	)
	parser.add_argument("--text-col", default="text", help="Column name for source text.")
	parser.add_argument("--response-col", default="response", help="Column name for model response.")
	parser.add_argument("--therapy-quality-col", default="therapy_quality", help="Column name for therapy quality score.")
	parser.add_argument("--identifies-belief-col", default="identifies_belief", help="Column name for belief identification score.")
	parser.add_argument("--asks-evidence-col", default="asks_evidence", help="Column name for evidence/alternative score.")
	parser.add_argument("--avoids-validation-col", default="avoids_validation", help="Column name for avoiding validation score.")
	parser.add_argument("--proposes-next-step-col", default="proposes_next_step", help="Column name for next-step score.")
	parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size per API call.")
	parser.add_argument("--max-retries", type=int, default=3, help="Max retries per batch.")
	parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep between retries.")
	return parser.parse_args()


def _parse_run_filename(filename: str) -> Optional[Tuple[str, str]]:
	base = os.path.basename(filename)
	if not base.endswith(".csv"):
		return None
	parts = base[:-4].split("_")
	if len(parts) < 5:
		return None
	if (parts[0] != "steering" and parts[0] != "random") or parts[1] != "layer" or parts[3] != "alpha":
		return None
	return parts[2], parts[4]


def _list_run_csvs(input_dir: str) -> List[str]:
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Input directory not found: {input_dir}")
	paths = []
	for name in os.listdir(input_dir):
		full_path = os.path.join(input_dir, name)
		if _parse_run_filename(name) and os.path.isfile(full_path):
			paths.append(full_path)
	return sorted(paths)


def main() -> None:
	args = parse_args()
	load_dotenv()
	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY is not set.")

	client = OpenAI()
	column_map = {
		"therapy_quality": args.therapy_quality_col,
		"identifies_belief": args.identifies_belief_col,
		"asks_evidence": args.asks_evidence_col,
		"avoids_validation": args.avoids_validation_col,
		"proposes_next_step": args.proposes_next_step_col,
	}

	if args.input_csv:
		if not args.output_csv:
			raise ValueError("--output-csv is required when --input-csv is used.")
		df = pd.read_csv(args.input_csv)
		for col in (args.text_col, args.response_col):
			if col not in df.columns:
				raise ValueError(f"Missing required column: {col}")

		scored = score_dataframe(
			df,
			client,
			args.model,
			args.text_col,
			args.response_col,
			column_map,
			args.batch_size,
			args.max_retries,
			args.retry_sleep,
		)
		scored.to_csv(args.output_csv, index=False)
		return

	os.makedirs(args.output_dir, exist_ok=True)
	input_paths = _list_run_csvs(args.input_dir)
	if not input_paths:
		raise FileNotFoundError(f"No run CSVs found in: {args.input_dir}")

	for input_path in tqdm(input_paths, desc="Run files", unit="file"):
		layer_alpha = _parse_run_filename(os.path.basename(input_path))
		if layer_alpha is None:
			continue
		layer, alpha = layer_alpha
		response_col = f"therapy_output_layer_{layer}_alpha_{alpha.split('p')[0]}.{alpha.split('p')[1]}"
		output_path = os.path.join(args.output_dir, os.path.basename(input_path))

		df = pd.read_csv(input_path)
		for col in (args.text_col, response_col):
			if col not in df.columns:
				raise ValueError(f"Missing required column: {col} in {input_path}")

		scored = score_dataframe(
			df,
			client,
			args.model,
			args.text_col,
			response_col,
			column_map,
			args.batch_size,
			args.max_retries,
			args.retry_sleep,
		)
		scored.to_csv(output_path, index=False)


if __name__ == "__main__":
	main()
