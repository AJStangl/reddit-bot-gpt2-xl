import base64
import hashlib
import logging
import os
import time
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

from core.components.text.services.generation_arguments import image_generation_arguments


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import enum


class Device(enum.Enum):
	gpu = "cuda"
	cpu = "cpu"


class TextGenerator:
	def __init__(self, model_path: str = os.environ.get("MODEL_PATH")):
		self.model_path: str = model_path
		self.text_model: GPT2LMHeadModel = self.load_model(self.model_path)
		self.tokenizer: GPT2Tokenizer = self.load_tokenizer(self.model_path)
		self.device: torch.device = torch.device("cuda")
		self.text_model.to(self.device)

	def set_device(self, device: Device):
		device: torch.device = torch.device(str(device.value))
		self.device = device

	def load_model(self, model_path: str) -> GPT2LMHeadModel:
		logger.info(f":: Loading GPT2 LM Head Model")
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		model.eval()
		return model

	def load_tokenizer(self, model_path: str) -> GPT2Tokenizer:
		logger.info(f":: Loading GPT2 Tokenizer")
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer

	@torch.no_grad()
	def generate(self, prompt: str) -> Optional[str]:
		try:
			encoding = self.tokenizer(prompt, padding=False, return_tensors='pt').to(self.device)
			inputs = encoding['input_ids']
			attention_mask = encoding['attention_mask']
			self.check_encoding(inputs=inputs, attention_mask=attention_mask)
			config = {
				'inputs': inputs,
				'attention_mask': attention_mask,
				'max_length': 512,
				'repetition_penalty': 1.1,
				'num_return_sequences': 1,
				'temperature': 1.2,
				'top_k': 50,
				'top_p': 0.95,
				'do_sample': True,
			}
			result = None
			for item in self.text_model.generate(**config):
				result = self.tokenizer.decode(item, skip_special_tokens=False, clean_up_tokenization_spaces=True)
				break
			return result

		except Exception as e:
			logger.error(e)
			exit(1)

	def check_encoding(self, inputs, attention_mask):
		if inputs.size(0) <= 0 or attention_mask.size(0) <= 0:
			logger.error("Inputs Fail: inputs.size(0) <= 0 or attention_mask.size(0) <= 0")
			raise ValueError("Inputs Fail: inputs.size(0) <= 0 or attention_mask.size(0) <= 0")
		if inputs.dim() != 2 or attention_mask.dim() != 2:
			logger.error("Invalid shape. Expected 2D tensor.")
			raise ValueError("Invalid shape. Expected 2D tensor.")
		if inputs.shape != attention_mask.shape:
			logger.error("Mismatched shapes between input_ids and attention_mask.")
			raise ValueError("Mismatched shapes between input_ids and attention_mask.")

	def get_generative_text_args(self, inputs, attention_mask) -> dict:
		args = {
			'input_ids': inputs,
			'attention_mask': attention_mask,
			'max_new_tokens': 512,
			'repetition_penalty': 1.1,
			'temperature': 1.2,
			'top_k': 50,
			'top_p': 0.95,
			'do_sample': True,
			'num_return_sequences': 1
		}
		return args


class GenerativeServices:
	def __init__(self):
		self.text_lock_path = os.path.join(os.environ.get("LOCK_PATH"), "text.lock")
		self.image_lock_path = os.path.join(os.environ.get("LOCK_PATH"), "sd.lock")
		self.detoxify: pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=torch.device("cpu"))
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
		self.text_generator: TextGenerator = TextGenerator()

	def get_image_from_standard_diffusion(self, caption: str) -> str:
		try:
			base_address = os.environ.get("STANDARD_DIFFUSION_URL")
			response = requests.get(base_address)
			if response.status_code != 200:
				return None

			endpoint = base_address + "/sdapi/v1/txt2img"
			data = image_generation_arguments(caption)

			headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
			data_json = data
			response = requests.post(endpoint, json=data_json, headers=headers)
			if response.status_code != 200:
				return None
			r = response.json()
			out_path = os.environ.get("IMAGE_OUT_DIR")
			os.makedirs(out_path, exist_ok=True)
			image_hash = None
			data = []
			for i, _ in enumerate(r['images']):
				image = Image.open(BytesIO(base64.b64decode(_.split(",", 1)[0])))
				image_hash = hashlib.md5(image.tobytes()).hexdigest()
				save_path = os.path.join(out_path, f'{image_hash}-{i}.png')
				image.save(save_path)
				data.append({
					'image_path': save_path,
					'caption': caption,
				})
				image.close()
			return data[0]['image_path']
		except Exception as e:
			logger.exception(e)
			return None

	def caption_image_from_url(self, image_url: str) -> str:
		result = ""
		try:
			response = requests.get(image_url)
			if response.status_code != 200:
				return ""
			content = response.content

			image = Image.open(BytesIO(content))
			try:
				self.model.to("cpu")
				inputs = self.processor(images=image, return_tensors="pt").to("cpu")
				out = self.model.generate(**inputs, max_new_tokens=77, num_return_sequences=1, do_sample=True)
				result = self.processor.decode(out[0], skip_special_tokens=True)
			except Exception as e:
				logger.exception(e)
				raise e
			finally:
				image.close()

		except Exception as e:
			logger.exception(e)
			result = ""
		finally:
			return result

	def get_info_string(self, prompt, completion):
		info_string = \
			f"""
		===================================
		Prompt: {prompt}
		Completion: {completion}
		===================================
		"""
		return info_string

	def create_prompt_completion(self, prompt: str) -> Optional[str]:
		try:
			start_time: time = time.time()
			completion: str = self.text_generator.generate(prompt=prompt)
			end_time: time = time.time()
			elapsed_time: float = end_time - start_time
			logger.info(f":: Time taken for run_generation: {elapsed_time:.4f} seconds")
			cleaned_completion: str = self.clean_text(completion=completion, prompt=prompt)
			logger.debug(self.get_info_string(prompt=prompt, completion=completion))
			return cleaned_completion
		except Exception as e:
			logger.exception(e)
			raise e

	def create_prompt_for_submission(self, prompt: str) -> Optional[dict]:
		try:
			start_time: time = time.time()
			completion: str = self.text_generator.generate(prompt=prompt)
			end_time: time = time.time()
			elapsed_time: float = end_time - start_time
			logger.info(f":: Time taken for run_generation: {elapsed_time:.4f} seconds")
			cleaned_completion: Optional[dict] = self.clean_completion_for_submission(completion=completion)
			if "<|image|>" in completion:
				path = self.get_image_from_standard_diffusion(completion)
				if path is not None:
					return {
						'title':  cleaned_completion.get("title"),
						'text': cleaned_completion.get("text"),
						'image': path,
					}
				else:
					return None

			if cleaned_completion.get("text") == "":
				path = self.get_image_from_standard_diffusion(caption=cleaned_completion.get("title"))
				if path is not None:
					return {
						'title':  cleaned_completion.get("title"),
						'text': cleaned_completion.get("text"),
						'image': path,
					}

			else:
				cleaned_completion: str = self.clean_completion_for_submission(completion=completion)
				logger.debug(self.get_info_string(prompt=prompt, completion=completion))
				return cleaned_completion
		except Exception as e:
			logger.exception(e)
			return None

	def clean_completion_for_submission(self, completion: str) -> Optional[dict]:
		import re
		pattern = r'<\|(?P<tag>.*?)\|>(?P<value>.*?)(?=(<\|)|$)'
		results = {
			'title': None,
			'text': None,
			'image': None,
		}
		completions = re.findall(pattern, completion)
		try:
			for item in completions:
				tag = item[0]
				value = item[1]
				if tag == 'title':
					results['title'] = value
				if tag == 'text':
					results['text'] = value
				if tag == 'image':
					results['image'] = value
			return results
		except Exception as e:
			logger.exception(e)
			return None

	def clean_text(self, completion, prompt) -> Optional[str]:
		try:
			replaced = completion.replace(prompt, "")
			split_target = replaced.split("<|context_level|>")
			if len(split_target) > 0:
				final = split_target[0].replace("<|endoftext|>", "")
				if self.ensure_non_toxic(final):
					return final
				else:
					return None
			else:
				# now we need to check if there is only an <|endoftext|>
				split_target = replaced.split("<|endoftext|>")
				if len(split_target) > 0:
					final = split_target[0].replace("<|endoftext|>", "")
					if self.ensure_non_toxic(final):
						return final
					else:
						return None
				else:
					return None
		except Exception as e:
			logger.error(e)
			return None

	def ensure_non_toxic(self, input_text: str) -> bool:
		try:
			threshold_map = {
				'toxic': 0.75,
				'obscene': 0.75,
				'insult': 0.75,
				'identity_attack': 0.75,
				'identity_hate': 0.75,
				'severe_toxic': 0.75,
				'threat': 1.0
			}
			results = self.detoxify.predict(input_text)[0]

			for key in threshold_map:
				label = results.get("label")
				score = results.get("score")
				if key == label:
					if score > threshold_map[key]:
						logging.info(f"Detoxify: {key} score of {score} is above threshold of {threshold_map[key]}")
						return False
				continue

			return True
		except Exception as e:
			logger.exception(e)
			return False
