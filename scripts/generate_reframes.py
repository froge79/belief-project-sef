import argparse
import csv
import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = (
	"You are generating a balanced cognitive reframe for a single text sample. "
	"A balanced reframe acknowledges the difficulty of the situation while avoiding extreme or absolute conclusions. It should introduce a more flexible and realistic interpretation without forcing positivity."
	"It should be in the same perspective as"
	"the original thought, not an outside observer. Act as the person themselves, not a therapist. "
	"Do not shame, invalidate, or dismiss feelings. Keep it concise, natural, and specific."
	"Also decide if the text is viable: discard=1 if the text is not depressive, irrelevant, incoherent, or too ambiguous to label; otherwise 0."
	"If discard=1, the reframe must be an empty string."
	"The text sample could be Nobody at school likes me; I'll always be alone."
	"The corresponding reframe could be I haven't connected with many people at school lately, but I have friends in other circles and I can work on reaching out more."
)


def _build_request_payload(batch, text_column):
	items = []
	for idx, row in enumerate(batch, start=1):
		items.append(
			{
				"index": idx,
				"text": row.get(text_column, ""),
			}
		)

	user_content = (
		"For each item, write a balanced reframe and decide discard. "
		"Return JSON with a single key 'reframes', an array in the same order. "
		"Each entry must have keys 'reframe' and 'discard'. "
		"If discard=1, reframe must be an empty string.\n\n"
		+ json.dumps(items, ensure_ascii=False)
	)

	return {
		"model": "gpt-4o-2024-08-06",
		"temperature": 0,
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "balanced_reframes",
				"strict": True,
				"schema": {
					"type": "object",
					"properties": {
						"reframes": {
							"type": "array",
							"items": {
								"type": "object",
								"properties": {
										"reframe": {"type": "string"},
										"discard": {"type": "integer"},
								},
									"required": ["reframe", "discard"],
								"additionalProperties": False,
							},
						},
					},
					"required": ["reframes"],
					"additionalProperties": False,
				},
			},
		},
		"messages": [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": user_content},
		],
	}


def _call_openai_api(client, payload, max_retries=5):
	for attempt in range(max_retries):
		try:
			return client.chat.completions.create(**payload)
		except Exception as exc:
			if attempt < max_retries - 1:
				time.sleep(2 ** attempt)
				continue
			raise exc


def _extract_reframes(response, expected_count):
	try:
		content = response.choices[0].message.content
		parsed = json.loads(content)
		reframes = parsed.get("reframes", [])
	except (KeyError, json.JSONDecodeError, TypeError):
		raise ValueError("Unexpected API response format")

	if not isinstance(reframes, list) or len(reframes) != expected_count:
		raise ValueError("API returned unexpected reframe count")

	for item in reframes:
		if not isinstance(item, dict) or "reframe" not in item or "discard" not in item:
			raise ValueError("API returned invalid reframe object")

	return reframes


def _truncate_text(text, max_words=200):
	words = str(text).split()
	if len(words) <= max_words:
		return str(text), 0
	return " ".join(words[:max_words]), 1


def _read_rows(input_path, text_column):
	with open(input_path, "r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		missing = {text_column} - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
		return list(reader), list(reader.fieldnames or [])


def _write_rows(output_path, rows, fieldnames):
	with open(output_path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main():
	parser = argparse.ArgumentParser(
		description="Generate balanced reframe statements using the OpenAI API."
	)
	parser.add_argument(
		"--input",
		default="data/phase4_dreaddit_800.csv",
		help="Path to input CSV.",
	)
	parser.add_argument(
		"--output",
		default="output/phase4_dreaddit_800_reframes.csv",
		help="Path to output CSV.",
	)
	parser.add_argument("--batch-size", type=int, default=5, help="Rows per API call.")
	parser.add_argument(
		"--text-column",
		default="text",
		help="Column containing distorted thoughts.",
	)
	parser.add_argument(
		"--reframe-column",
		default="reframe",
		help="Column name to write reframes into.",
	)
	parser.add_argument(
		"--max-words",
		type=int,
		default=200,
		help="Max words of input text per item.",
	)
	args = parser.parse_args()

	load_dotenv()

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY is not set.")

	client = OpenAI(api_key=api_key)

	rows, input_fieldnames = _read_rows(args.input, args.text_column)
	processed_rows = []

	batch_size = max(1, args.batch_size)
	total_batches = (len(rows) + batch_size - 1) // batch_size

	for start in tqdm(range(0, len(rows), batch_size), total=total_batches, desc="Batches"):
		batch = rows[start : start + batch_size]
		prepared_batch = []
		for row in batch:
			truncated_text, truncated_flag = _truncate_text(
				row.get(args.text_column, ""), args.max_words
			)
			prepared_batch.append(
				{
					**row,
					args.text_column: truncated_text,
					"truncated (0/1)": truncated_flag,
				}
			)

		payload = _build_request_payload(prepared_batch, args.text_column)
		response = _call_openai_api(client, payload)
		reframes = _extract_reframes(response, len(prepared_batch))

		for row, reframe in zip(prepared_batch, reframes):
			output_row = dict(row)
			discard = int(reframe.get("discard", 0))
			output_row["discard"] = discard
			output_row[args.reframe_column] = reframe["reframe"] if discard == 0 else ""
			processed_rows.append(output_row)

	output_fieldnames = list(input_fieldnames)
	if args.reframe_column not in output_fieldnames:
		output_fieldnames.append(args.reframe_column)
	if "discard" not in output_fieldnames:
		output_fieldnames.append("discard")
	if "truncated (0/1)" not in output_fieldnames:
		output_fieldnames.append("truncated (0/1)")

	_write_rows(args.output, processed_rows, output_fieldnames)


if __name__ == "__main__":
	main()
