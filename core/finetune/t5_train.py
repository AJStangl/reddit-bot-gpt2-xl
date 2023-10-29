import csv
from typing import List, Dict
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


class RedditT5Dataset:
	def __init__(self, file_path: str):
		self.file_path = file_path
		self.dataset = self.create_t5_finetuning_dataset()

	@staticmethod
	def read_file(file_path: str) -> str:
		with open(file_path, 'r', encoding='utf-8') as f:
			return f.read()

	@staticmethod
	def parse_reddit_submission(file_content: str) -> List[Dict[str, str]]:
		entries = file_content.split('<|endoftext|>')
		parsed_entries = []
		for entry in entries:
			if not entry.strip():
				continue
			parsed_entry = {}
			for field in ['subreddit', 'title', 'text', 'context_level', 'comment']:
				start_token = f'<|{field}|>'
				end_token = '<|' if field != 'comment' else '<|endoftext|>'
				start_idx = entry.find(start_token)
				if start_idx == -1:
					continue
				start_idx += len(start_token)
				end_idx = entry.find(end_token, start_idx)
				parsed_entry[field] = entry[start_idx:end_idx].strip()
			parsed_entries.append(parsed_entry)
		return parsed_entries

	def create_t5_finetuning_dataset(self) -> List[Dict[str, str]]:
		content = self.read_file(self.file_path)
		parsed_entries = self.parse_reddit_submission(content)
		t5_data = []

		# Initialize with subreddit, title, and text
		input_str = f"subreddit: {parsed_entries[0].get('subreddit', '')} title: {parsed_entries[0].get('title', '')} text: {parsed_entries[0].get('text', '')}"

		for i in range(len(parsed_entries)):
			input_entry = parsed_entries[i]

			# Append the comment and its context level to input_str
			input_str += f" context_level: {input_entry.get('context_level', '')} comment: {input_entry.get('comment', '')}"

			# Look ahead for a comment at the next context level
			for j in range(i + 1, len(parsed_entries)):
				output_entry = parsed_entries[j]
				if int(output_entry.get('context_level', -1)) == int(input_entry.get('context_level', 0)) + 1:
					output_str = f"context_level: {output_entry.get('context_level', '')} comment: {output_entry.get('comment', '')}"
					t5_data.append({'input': input_str, 'output': output_str})
					# Stop looking ahead once we find a comment at the next context level
					break

		return t5_data

	def save_to_csv(self, train_file_path, validation_file_path, split_ratio=0.8):
		# Calculate number of training samples
		num_train = int(len(self.dataset) * split_ratio)

		# Split dataset into training and validation sets
		train_data = self.dataset[:num_train]
		validation_data = self.dataset[num_train:]

		# Write training data to CSV
		with open(train_file_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['input', 'output'])  # header
			for item in train_data:
				writer.writerow([item['input'], item['output']])

		# Write validation data to CSV
		with open(validation_file_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['input', 'output'])  # header
			for item in validation_data:
				writer.writerow([item['input'], item['output']])


class Tech:
	@staticmethod
	def create_t5_fine_tuning_dataset(file_content):
		parsed_entries = RedditT5Dataset.parse_reddit_submission(file_content)
		t5_data = []

		# Initialize with subreddit, title, and text, properly separated
		input_str = f"subreddit: {parsed_entries[0].get('subreddit', '')} title: {parsed_entries[0].get('title', '')} text: \"{parsed_entries[0].get('text', '')}\""

		# Debug variables to track the flow
		debug_info = {
			'total_entries': len(parsed_entries),
			'input_str_initial': input_str,
			'output_str_found': 0,
			'loops_executed': 0
		}

		for i in range(1, len(parsed_entries)):
			input_entry = parsed_entries[i]

			# Append the comment and its context level to input_str
			input_str += f" context_level: {input_entry.get('context_level', '')} comment: {input_entry.get('comment', '')}"

			debug_info['loops_executed'] += 1  # Increment loop counter

			# Look ahead for a comment at the next context level
			for j in range(i + 1, len(parsed_entries)):
				output_entry = parsed_entries[j]
				if int(output_entry.get('context_level', -1)) == int(input_entry.get('context_level', 0)) + 1:
					output_str = f"context_level: {output_entry.get('context_level', '')} comment: {output_entry.get('comment', '')}"
					t5_data.append({'input': input_str, 'output': output_str})

					debug_info['output_str_found'] += 1  # Increment output string counter

					break  # Stop looking ahead once we find a comment at the next context level

			return t5_data, debug_info





if __name__ == '__main__':
	content = RedditT5Dataset.read_file("D:\\code\\repos\\reddit-bot-gpt2-xl\\core\\finetune\\data\\17h66vt.txt")
	data, info = Tech.create_t5_fine_tuning_dataset(content)
	print(data, info)
	# reddit_dataset.save_to_csv('train.csv', 'validation.csv')
