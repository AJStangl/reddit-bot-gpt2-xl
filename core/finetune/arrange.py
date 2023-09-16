import os
import random
from datetime import datetime
from pathlib import Path


def combine_text_files(directory, output_file):
	with open(output_file, 'wb') as outfile:
		for file in os.listdir(directory):
			file_path = os.path.join(directory, file)
			if file_path.endswith('.txt'):
				with open(file_path, 'rb') as infile:
					file_data = infile.readlines()
					random.shuffle(file_data)
					for line in file_data:
						line = line.decode('unicode_escape')
						if line.__contains__('[deleted]') or line.__contains__('[removed]'):
							continue
						else:
							line = line.encode('unicode_escape')
							outfile.write(line)
							outfile.write(b'\n')


def main():
	out_file_path = f"raw_training_{datetime.now().timestamp()}.txt"

	script_dir = Path(__file__).resolve().parent
	data_path = Path(script_dir, "data")
	combine_text_files(data_path, out_file_path)

	with open(out_file_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		print(f"Total Lines: {len(lines)}")
		print(f"Total Characters: {sum([len(line) for line in lines])}")
		print(f"Average Characters: {sum([len(line) for line in lines]) / len(lines)}")
		print(f"Average Tokens: {sum([len(line.split()) for line in lines]) / len(lines)}")
