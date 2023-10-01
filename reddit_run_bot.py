import json
import logging
import os
import random
import re
import threading
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from queue import Queue
from typing import Optional

import praw
import prawcore
import requests
import torch
from PIL import Image
from dotenv import load_dotenv
from praw.models import Comment, Submission
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlipProcessor, BlipForConditionalGeneration
from transformers import logging as transformers_logging
from transformers import pipeline

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity(transformers_logging.FATAL)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CaptionProcessor:
	def __init__(self, device_name: str = "cuda"):
		self.device_name = device_name
		self.device = torch.device(self.device_name)
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

	def caption_image_from_url(self, image_url: str) -> str:
		result = ""
		try:
			response = requests.get(image_url)
			if response.status_code != 200:
				return ""
			content = response.content

			image = Image.open(BytesIO(content))
			try:
				self.model.to(self.device)
				inputs = self.processor(images=image, return_tensors="pt").to(self.device)
				out = self.model.generate(**inputs, max_new_tokens=77, num_return_sequences=1, do_sample=True)
				result = self.processor.decode(out[0], skip_special_tokens=True)
			except Exception as e:
				logger.exception(e)
				result = ""
			finally:
				image.close()

		except Exception as e:
			logger.exception(e)
			result = ""
		finally:
			return result


class ModelRunner:
	def __init__(self, model_path: str):
		self.required_model_load = True
		self.model_path: str = model_path
		self.device: torch.device = torch.device('cuda')
		self.tokenizer: GPT2Tokenizer = self.load_tokenizer(self.model_path)
		self.detoxify: pipeline = pipeline("text-classification", model="unitary/toxic-bert",
										   device=torch.device("cpu"))
		self.caption_processor: CaptionProcessor = CaptionProcessor("cpu")
		self.text_model: GPT2LMHeadModel = self.load_model(self.model_path)
		self.text_model.to(self.device)
		self.text_lock_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\locks\\text.lock"
		self.image_lock_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\locks\\sd.lock"

	def create_lock(self):
		try:
			with open(self.text_lock_path, "wb") as handle:
				handle.write(b"")
		except Exception as e:
			logging.error(f"An error occurred while creating temp lock: {e}")

	def clear_lock(self):
		try:
			if os.path.exists(self.text_lock_path):
				os.remove(self.text_lock_path)
			else:
				logging.warning(f"Lock file {self.text_lock_path} does not exist.")
		except Exception as e:
			logging.error(f"An error occurred while deleting text lock: {e}")

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

	def run(self, text):
		while self.is_in_lock_state():
			continue
		encoding = self.tokenizer(text, padding=True, return_tensors='pt', truncation=True)
		if len(encoding) > 512:
			logger.info(f"The encoding output {len(encoding)} > 512, not performing operation.")
			return None
		encoding.to(self.device)
		try:
			return self.run_generation(encoding)
		except Exception as e:
			logger.error(e)
			raise Exception("I blew the fuck up exception", e)

	def is_in_lock_state(self):
		if os.path.exists(self.text_lock_path):
			return True
		if os.path.exists(self.image_lock_path):
			return True
		else:
			return False

	@torch.no_grad()
	def run_generation(self, encoding) -> str:
		try:
			start_time = time.time()
			self.create_lock()
			inputs = encoding['input_ids']
			attention_mask = encoding['attention_mask']
			if inputs.size(0) <= 0 or attention_mask.size(0) <= 0:
				logger.error("Inputs Fail: inputs.size(0) <= 0 or attention_mask.size(0) <= 0")
				return ""
			if inputs.dim() != 2 or attention_mask.dim() != 2:
				logger.error("Invalid shape. Expected 2D tensor.")
				return ""
			if inputs.shape != attention_mask.shape:
				logger.error("Mismatched shapes between input_ids and attention_mask.")
				return ""

			args = {
				'inputs': inputs,
				'attention_mask': attention_mask,
				'max_new_tokens': 512,
				'repetition_penalty': 1.1,
				'temperature': 1.2,
				'top_k': 50,
				'top_p': 0.95,
				'do_sample': True,
				'num_return_sequences': 1
			}
			temp = ""

			logger.info(":: Starting Text Generation")
			for i, _ in enumerate(self.text_model.generate(**args)):
				generated_texts = self.tokenizer.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=True)
				generated_texts = generated_texts.split("<|startoftext|>")
				good_line = ""
				for line in generated_texts:
					good_line = line

				temp = "<|startoftext|>" + good_line

			end_time = time.time()
			elapsed_time = end_time - start_time
			logger.info(f":: Time taken for run_generation: {elapsed_time:.4f} seconds")
			return temp

		except Exception as e:
			logger.info(e)
			exit(1)
		finally:
			self.clear_lock()

	def clean_text(self, text, input_string) -> Optional[str]:
		try:
			replaced = text.replace(input_string, "")
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

	def split_token_first_comment(self, prompt, completion) -> Optional[str]:
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
		threshold_map = {
			'toxic': 0.99,
			'obscene': 0.99,
			'insult': 0.99,
			'identity_attack': 0.99,
			'identity_hate': 0.99,
			'severe_toxic': 0.99,
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


class BotRunner:
	def __init__(self):
		self.queue_size = 0
		self.model_runner = ModelRunner(model_path=os.environ.get("MODEL_PATH"))
		self.bot_map: dict = self.read_bot_configuration()
		self.next_hour_current_time = 0
		self.reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
		self.task_queue = Queue()
		self.result_queue = Queue()

	def responding_background_process(self):
		retry_count = 0
		max_retries = 3

		logger.info(":: Starting responding_background_process")
		while True:
			if self.result_queue.empty():
				time.sleep(1)
				continue

			input_data: dict = self.result_queue.get()
			reply_id = input_data.get("reply_id")
			new_reddit = None
			try:
				reply_text = input_data.get("text")
				reply_bot = input_data.get("responding_bot")
				reply_sub = input_data.get("subreddit")
				reply_type = input_data.get("type")

				if reply_text is None or reply_text == "":
					self.result_queue.task_done()
					retry_count = 0
					continue

				new_reddit = praw.Reddit(site_name=reply_bot)

				if reply_type == 'submission':
					submission = new_reddit.submission(reply_id)
					reply = submission.reply(reply_text)
					logger.info(f":: {reply_bot} has replied to Submission: at https://www.reddit.com{reply.permalink}")
					self.result_queue.task_done()
					retry_count = 0

				if reply_type == 'comment':
					comment = new_reddit.comment(reply_id)
					reply = comment.reply(reply_text)
					logger.info(f":: {reply_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")
					self.result_queue.task_done()
					retry_count = 0

			except prawcore.PrawcoreException as e:
				logger.error(f"APIException: {e}")
				if retry_count < max_retries:
					retry_count += 1
					time.sleep(2 ** retry_count)
					self.result_queue.put(input_data)
				else:
					logger.error(f"Max retries reached for comment {reply_id}")

			except Exception as e:
				logger.error(f"An unexpected error occurred: {e}")

	def task_queue_handler(self, data: dict):
		self.task_queue.put(data)

	def text_generation_background_task(self):
		retry_count = 0
		max_retries = 3

		logger.info(":: Starting text_generation_background_task")
		while True:
			if self.task_queue.empty():
				time.sleep(1)
				continue

			input_data: dict = self.task_queue.get()
			input_string = input_data.get('text')
			try:
				text = self.model_runner.run(input_string)
				if text is None:
					logger.info(f":: Text is None, skipping.")
					self.task_queue.task_done()
					continue
				if input_data.get("type") == 'submission':
					reply_text = self.model_runner.split_token_first_comment(prompt=input_string, completion=text)
				else:
					reply_text = self.model_runner.clean_text(input_string=input_string, text=text)
				input_data['text'] = reply_text

				self.result_queue.put(input_data)
				self.task_queue.task_done()
				retry_count = 0

			except Exception as e:
				logger.error(f"An error occurred: {e}")

				if str(e) == "I blew the fuck up exception":
					logger.critical("Critical exception from `run` method. Exiting program.")
					exit(1)

				if retry_count < max_retries:
					retry_count += 1
					time.sleep(2 ** retry_count)
					self.task_queue.put(input_data)
				else:
					logger.error(f"Max retries reached for input {input_string}")

			time.sleep(1)

	def get_value_by_key(self, key, filename='cache.json'):
		existing_data = self.load_dict_from_file(filename)
		return existing_data.get(key, None)

	def set_value_by_key(self, key, value, filename='cache.json'):
		existing_data = self.load_dict_from_file(filename)
		data = {key: value}
		existing_data.update(data)

	def load_dict_from_file(self, filename='cache.json', max_retries=3, retry_delay=1):
		for attempt in range(max_retries):
			try:
				with open(filename, 'r') as f:
					return json.load(f)
			except FileNotFoundError:
				return {}
			except Exception as e:
				logger.error(f"Failed to load JSON from {filename}, attempt {attempt + 1}: {e}")
				time.sleep(retry_delay)
		logger.error(f"Failed to load JSON from {filename} after {max_retries} attempts.")
		return {}

	def read_bot_configuration(self) -> dict:
		bot_map = {}
		with open(os.environ.get("CONFIG_PATH"), 'r') as f:
			config = json.load(f)
			for item in config:
				bot_map[item['name']] = item['personality']
		return bot_map

	def construct_context_string(self, comment: Comment) -> str:
		things = []
		current_comment = comment
		counter = 0
		try:
			while not isinstance(current_comment, praw.models.Submission):
				if counter == 12:
					break
				comment_key = str(current_comment.id) + "-" + 'text'

				cached_thing = self.get_value_by_key(comment_key)
				if cached_thing is not None:
					cached = {
						'counter': counter,
						'text': cached_thing
					}
					things.append(cached)
					counter += 1
					current_comment = current_comment.parent()
					continue
				else:
					thing = {
						"text": "", "counter": 0
					}
					if thing is None:
						current_comment = current_comment.parent()
						time.sleep(1)
						continue
					thing['counter'] = counter
					thing['text'] = current_comment.body
					things.append(current_comment.body)
					self.set_value_by_key(comment_key, thing['text'])
					counter += 1
					current_comment = current_comment.parent()
					time.sleep(1)
					continue
		except prawcore.exceptions.RequestException as request_exception:
			logger.exception("Request Error", request_exception)
		except Exception as e:
			logger.exception(f"General Exception In construct_context_string", e)

		things.reverse()
		out = ""
		for i, r in enumerate(things):
			out += f"<|context_level|>{i}<|comment|>{r}"

		out += f"<|context_level|>{len(things)}<|comment|>"
		return out

	def create_post_string(self) -> dict:
		chosen_bot_key = random.choice(list(self.bot_map.keys()))
		bot_config = self.bot_map[chosen_bot_key]
		constructed_string = f"<|startoftext|><|subreddit|>r/{bot_config}"
		result = self.model_runner.run(constructed_string)

		pattern = re.compile(r'<\|([a-zA-Z0-9_]+)\|>(.*?)(?=<\|[a-zA-Z0-9_]+\|>|$)', re.DOTALL)
		matches = pattern.findall(result)
		result_dict = {key: value for key, value in matches}
		return {
			'title': result_dict.get('title'),
			'text': result_dict.get('text'),
			'image_path': None,
			'bot': chosen_bot_key,
			'subreddit': os.environ.get("SUBREDDIT_TO_MONITOR")
		}

	def create_reddit_post(self, data: dict) -> None:
		bot = data.get("bot")
		subreddit_name = data.get("subreddit")
		new_reddit = praw.Reddit(site_name=bot)
		create_image = random.choice([False, False])
		try:
			subreddit = new_reddit.subreddit(subreddit_name)
			title = data.get("title")
			text = data.get('text')
			if create_image:
				return
			else:
				result = subreddit.submit(title, selftext=text)
				logger.info(f"{bot} has Created A Submission: at https://www.reddit.com{result.permalink}")
		except Exception as e:
			logger.error(e)
			raise e

	def process_submission(self, submission):
		if submission is None:
			time.sleep(1)
			return

		if str(submission.url).endswith(('.png', '.jpg', '.jpeg')):
			logger.debug(f":: Submission does not contain image URL: {submission.url}")
			text = self.model_runner.caption_processor.caption_image_from_url(submission.url)

		else:
			logger.debug(f":: Submission contains image URL: {submission.url}")
			text = submission.selftext
		for bot in self.bot_map.keys():
			if str(submission.author).lower() == bot.lower():
				continue
			personality = self.bot_map[bot]
			mapped_submission = {
				"subreddit": 'r/' + personality,
				"title": submission.title,
				"text": text
			}

			constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}<|context_level|>0<|comment|>"
			bot_reply_key = f"{bot}_{submission.id}"
			bot_reply_value = self.get_value_by_key(bot_reply_key)
			if bot_reply_value:
				continue

			data = {
				'text': constructed_string,
				'responding_bot': bot,
				'subreddit': mapped_submission['subreddit'],
				'reply_id': submission.id,
				'type': 'submission'
			}
			self.set_value_by_key(bot_reply_key, True)
			self.task_queue_handler(data)

	def process_comment(self, comment: Comment):
		if comment is None:
			return

		submission_id = comment.submission
		bots = list(self.bot_map.keys())
		filtered_bot = [x for x in bots if x.lower() != str(comment.author).lower()]
		responding_bot = random.choice(filtered_bot)
		personality = self.bot_map[responding_bot]
		submission =  self.reddit.submission(submission_id)
		mapped_submission = {
			"subreddit": 'r' + '/' + personality,
			"title": submission.title,
			"text": submission.selftext
		}

		if int(submission.num_comments) > int(os.environ.get('MAX_REPLIES')):
			logger.debug(f":: Comment Has More Than 250 Replies, Skipping")
			self.set_value_by_key(comment.id, True)
			return

		constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
		constructed_string += self.construct_context_string(comment)
		data = {
			'text': constructed_string,
			'responding_bot': responding_bot,
			'subreddit': mapped_submission['subreddit'],
			'reply_id': comment.id,
			'type': 'comment'
		}
		self.task_queue_handler(data)
		self.set_value_by_key(comment.id, True)

	def reply_to_comment(self, comment: Comment, responding_bot: str, reply_text: str):
		new_reddit = praw.Reddit(site_name=responding_bot)
		try:
			comment: Comment = new_reddit.comment(comment.id)
			reply = comment.reply(reply_text)
			logger.info(f":: {responding_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")
		except Exception as e:
			logger.error(e)


	def run(self):
		responding_thread = threading.Thread(target=self.responding_background_process, name="RespondingThread")
		text_generation_background_thread = threading.Thread(target=self.text_generation_background_task, name="TextGenerationThread")

		responding_thread.start()
		text_generation_background_thread.start()

		sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		subreddit = self.reddit.subreddit(sub_names)

		count = 0
		logger.info(":: Starting Warm Up For Model Text Generation")
		self.model_runner.run("<|startoftext|><|subreddit|>r/test<|title|>I Need to test the love inside me<|text|>Please Come And Help Me<|context_level|>0<|comment|>")
		logger.info(f":: Starting Primary Loop, Iteration: {count}")
		while True:
			try:
				for item in subreddit.stream.comments(pause_after=-1, skip_existing=True):
					logger.debug(f":: Iteration: {count}, Item: {item}")
					if datetime.timestamp(datetime.now()) > self.next_hour_current_time:
						self.next_hour_current_time = datetime.timestamp(datetime.now() + timedelta(hours=3))
						self.set_value_by_key('next_time_to_post', self.next_hour_current_time)
						data = self.create_post_string()
						self.create_reddit_post(data)
					else:
						logger.debug(f":: Next Post In {self.next_hour_current_time - datetime.timestamp(datetime.now())} Seconds")

					if count % 10 == 0:
						logger.debug(":: Checking For Submissions")
						for x in subreddit.new(limit=5):
							if isinstance(x, Submission):
								time.sleep(1)
								if x is None:
									continue
								self.process_submission(submission=x)
								continue

					if item is None:
						count += 1
						continue

					comment_key = f"{item.id}-comment"
					comment_seen = self.get_value_by_key(comment_key)
					if comment_seen:
						count += 1
						continue

					if isinstance(item, Comment):
						self.process_comment(comment=item)
						self.set_value_by_key(comment_key, True)
						count += 1

			except Exception as e:
				if e.args[0] == "I blew the fuck up exception":
					logger.info(e)
					raise e
				else:
					logger.info("An error has occurred during the primary loop", e)
					time.sleep(1)
					count += 1
