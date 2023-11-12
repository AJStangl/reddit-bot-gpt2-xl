import os
import re
from tqdm import tqdm


def process_files(path, batch_size=1000):
	known_set = set()
	regex = re.compile(r"<\|startoftext\|><\|subreddit\|>(.*)<\|title\|>(.*)<\|text\|>(.*)")

	for root, dirs, files in os.walk(path):
		# Wrap the file processing loop with tqdm for progress monitoring
		for file in tqdm(files, desc="Processing files"):
			if file.endswith('.txt'):
				filepath = os.path.join(root, file)
				with open(filepath, 'r') as fin:
					batch = []
					for line in fin:
						batch.append(line)
						if len(batch) >= batch_size:
							process_batch(batch, regex, known_set)
							batch = []
					if batch:  # Process any remaining lines in the last batch
						process_batch(batch, regex, known_set)
	return known_set


def process_batch(batch, regex, known_set):
	for line in batch:
		match = regex.search(line)
		if match:
			found = match.group(1)
			if found not in known_set:
				known_set.add(found)



if __name__ == '__main__':
	path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\core\\finetune\\data\\"
	result = process_files(path)
	with open('topics.txt', 'w') as f:
		for item in result:
			f.write("%s\n" % item)
