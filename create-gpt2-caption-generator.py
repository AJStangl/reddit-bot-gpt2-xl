import json
import re
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from tqdm import tqdm
from torch.utils.data import Dataset, random_split


class SimpleGPTDataset(Dataset):
	_input_id: str = 'input_ids'
	_attention_mask: str = 'attention_mask'

	def __init__(self, encoded_data_set, _tokenizer, _max_length, truncation=True):
		self.input_ids = []
		self.attention_mask = []
		self.labels = []
		with tqdm(total=len(encoded_data_set)) as pbar:
			for encodings_dict in encoded_data_set:
				self.input_ids.append(torch.tensor(encodings_dict[self._input_id]))
				self.attention_mask.append(torch.tensor(encodings_dict[self._attention_mask]))
				pbar.update(1)

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, index):
		return self.input_ids[index], self.attention_mask[index]


@dataclass
class SubjectData:
	title: str
	caption: str
	subject: str

	def __str__(self):
		text = f"<|startoftext|><|model|>{self.subject}<|title|>{self.title}<|caption|>{self.caption}<|endoftext|>"
		text = text.encode("utf-8")
		return text.decode("utf-8")


def mask_social_text(text):
	masked_text = re.sub(r'#\w+', '', text)
	masked_text = re.sub(r'@\w+', '', masked_text)
	masked_text = masked_text.strip()
	split = masked_text.split("\n")
	masked_text = next(iter(split))
	if len(masked_text) == 0:
		masked_text = "Untitled"
	return masked_text


def get_tokenizer():
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	special_tokens_dict = {
		"bos_token": "<|startoftext|>",
		"eos_token": "<|endoftext|>",
		"pad_token": "<|pad|>",
		"unk_token": "<|unk|>",
		"additional_special_tokens": [
			"<|model|>",
			"<|title|>",
			"<|caption|>"
		]
	}
	tokenizer.add_special_tokens(special_tokens_dict)
	return tokenizer


def tokenizer_encode(tokenizer: GPT2Tokenizer, input_string: str) -> dict:
	return tokenizer(input_string, return_tensors="pt", padding=True, add_special_tokens=True)


def tokenizer_decode(tokenizer: GPT2Tokenizer, encoding):
	return tokenizer.decode(next(iter(encoding)), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def get_small_model(tokenizer: GPT2Tokenizer):
	model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
	model.resize_token_embeddings(len(tokenizer))
	return model


def train_data(dataset, tokenizer, model):
	model.cuda()
	model_output_dir = "prompt-model"

	generator = torch.Generator()

	generator.manual_seed(0)

	train_size = int(0.9 * len(dataset))

	train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)

	training_args = TrainingArguments(output_dir="prompt-model")
	training_args.num_train_epochs = 1
	training_args.per_device_train_batch_size = 1
	training_args.per_device_eval_batch_size = 1
	training_args.save_steps = 1000
	training_args.weight_decay = 0.0
	training_args.fp16 = True
	training_args.tf32 = True
	training_args.auto_find_batch_size = True
	training_args.gradient_accumulation_steps = 5
	training_args.learning_rate = 5e-5
	training_args.save_total_limit = 1
	training_args.overwrite_output_dir = True

	trainer: Trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=lambda x: {
			'input_ids': torch.stack([x[0] for x in x]),
			'attention_mask': torch.stack([x[1] for x in x]),
			'labels': torch.stack([x[0] for x in x])
		}
	)

	trainer.train()
	trainer.save_model(model_output_dir)


white_list = [
 'KoreanHotties',
 'hotofficegirls',
 'HotGirlNextDoor',
 'amihot',
 'sexygirls',
 'prettyasiangirls',
 'SFWNextDoorGirls',
 'OldLadiesBakingPies',
 'marleybrinxy',
 'RealGirls_SFW',
 'naughtynianacci',
 'mildlypenis',
 'AmIhotAF',
 'Ifyouhadtopickone',
 'princesskatiebeth',
 'bundleofbrittany',
 'AsianInvasion',
 'miakhalifa',
 'tightdresses',
 'celebrities',
 'greentext',
 'evolutionofevie',
 'PrettyGirls',
 'TrueFMK',
 'itookapicture',
 'sarameikasai',
 'WhitePeopleTwitter',
 'DressesPorn',
 'memes',
 'redheadsweetheart_',
 'blondebeachvibes',
 'EarthPorn',
 'CityPorn',
 'SFWRedheads',
 'AesPleasingAsianGirls',
 'DLAH',
 'bathandbodyworks',
 'realasians',
 'heytegan',
 'Dresses',
 'Faces',
 'CollaredDresses',
 'selfies',
 'sashagreyonlyfans',
 'trippinthroughtime',
 'Amicute',
 'fatsquirrelhate',
 'AsianOfficeLady',
 'sfwpetite',
 'WomenInLongDresses',
 'secret.sophie96',
 'SlitDresses',
 'gentlemanboners',
 'ellyclutchh',
 'wallstreetbets',
 'blairwears'
]




if __name__ == '__main__':
	path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\data\\captions.json"
	with open(path, 'r') as f:
		data = json.load(f)

	tokenizer_instance = get_tokenizer()

	all_subjects = []
	for partition in data:
		subject: str = next(iter(partition))
		if subject not in white_list:
			continue


		row_data: dict = partition[subject]
		for row in row_data:
			title = row['title']
			title = mask_social_text(title)
			captions = row['captions']
			for caption in captions:
				subject_data: SubjectData = SubjectData(title=title, caption=caption, subject=subject)
				all_subjects.append(subject_data)

	encoding_data = []

	with open('training.txt', 'wb') as f:
		for subject_data in all_subjects:
			tokenized_encoding = tokenizer_encode(tokenizer_instance, str(subject_data))
			encoding_data.append(tokenized_encoding)
			decoded = tokenizer_decode(tokenizer_instance, tokenized_encoding['input_ids'])
			f.write(decoded.encode("utf-8") + b"\n")


	dataset_instance = SimpleGPTDataset(encoding_data, tokenizer_instance, 1024, True)

	model_instance = get_small_model(tokenizer_instance)

	train_data(dataset_instance, tokenizer_instance, model_instance)




