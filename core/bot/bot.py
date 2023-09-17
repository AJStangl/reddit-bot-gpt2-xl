import json
import logging
import os
from io import BytesIO
from typing import Optional
import random
import aiohttp
import asyncpraw
import torch
from asyncpraw.models import Comment
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
from transformers import logging as transformers_logging
from transformers import pipeline

transformers_logging.set_verbosity(transformers_logging.FATAL)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import asyncio
import shelve


class CaptionProcessor(object):
	def __init__(self, device_name: str = "cuda"):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.device_name = device_name
		self.device = torch.device(self.device_name)
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

	async def caption_image_from_url(self, image_url: str) -> str:
		result = ""
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(image_url) as response:
					if response.status != 200:
						return ""
					content = await response.read()
					image = Image.open(BytesIO(content))
					try:
						inputs = self.processor(images=image, return_tensors="pt").to(self.device)
						out = self.model.generate(**inputs, max_new_tokens=77, num_return_sequences=1, do_sample=True)
						result = self.processor.decode(out[0], skip_special_tokens=True)
					except Exception as e:
						logger.exception(e)
						result = ""
					finally:
						response.close()
						image.close()
						await session.close()
		except Exception as e:
			logger.exception(e)
			result = ""
		finally:
			return result


class ModelRunner:
	def __init__(self, model_path):
		self.model_path = model_path
		self.device = torch.device('cuda')
		self.tokenizer, self.model = self.load_model_and_tokenizer(self.model_path)
		self.model.to(self.device)
		self.detoxify = pipeline("text-classification", model="unitary/toxic-bert", device=self.device)

	def run(self, text) -> str:
		encoding = self.get_encoding(text)
		encoding.to(self.device)
		output = self.run_generation(encoding)
		return output

	def get_encoding(self, text):
		encoding = self.tokenizer(text, padding=True, return_tensors='pt')
		return encoding

	@torch.no_grad()
	def run_generation(self, encoding):
		inputs = encoding['input_ids']
		attention_mask = encoding['attention_mask']
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
		logging.getLogger("transformers").setLevel(logging.FATAL)
		for i, _ in enumerate(self.model.generate(**args)):
			generated_texts = self.tokenizer.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=True)
			generated_texts = generated_texts.split("<|startoftext|>")
			good_line = ""
			for line in generated_texts:
				good_line = line

			temp = "<|startoftext|>" + good_line
			return temp
			# if self.ensure_non_toxic(temp): TODO: Figure out the tox stuff.
			# 	return temp
			# else:
			# 	logger.info("Output was Toxic")
			# 	self.run_generation(encoding)

	@staticmethod
	def clean_text(text, input_string) -> Optional[str]:
		try:
			replaced = text.replace(input_string, "")
			split_target = replaced.split("<|context_level|>")
			if len(split_target) > 0:
				final = split_target[0].replace("<|endoftext|>", "")
				return final
			else:
				# now we need to check if there is only an <|endoftext|>
				split_target = replaced.split("<|endoftext|>")
				if len(split_target) > 0:
					final = split_target[0].replace("<|endoftext|>", "")
					return final
				else:
					return None
		except Exception as e:
			logger.error(e)
			return None

	@staticmethod
	def split_token_first_comment(prompt, completion) -> Optional[str]:
		try:
			replaced = completion.replace(prompt, "")
			split_target = replaced.split("<|context_level|>")
			if len(split_target) > 0:
				final = split_target[0].replace("<|endoftext|>", "")
				return final
			else:
				# now we need to check if there is only an <|endoftext|>
				split_target = replaced.split("<|endoftext|>")
				if len(split_target) > 0:
					final = split_target[0].replace("<|endoftext|>", "")
					return final
				else:
					return None
		except Exception as e:
			logger.error(e)
			return None

	def load_model_and_tokenizer(self, model_path: str) -> (GPT2Tokenizer, GPT2LMHeadModel):
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer, model

	def ensure_non_toxic(self, input_text: str) -> bool:
		threshold_map = {
			'toxic': 0.99,
			'obscene':  0.99,
			'insult': 0.99,
			'identity_attack': 0.99,
			'identity_hate':  0.99,
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


class RedditRunner(object):
	def __init__(self):
		self.cache_path = os.path.join("cache", "cache")
		self.bot_map = self.set_bot_configration()
		logger.info(":: Initializing Model")
		self.model_runner: ModelRunner = ModelRunner(os.environ.get("MODEL_PATH"))
		logger.info(":: Initializing Captioning Model")
		self.caption_processor: CaptionProcessor = CaptionProcessor()
		logger.info(":: Initializing Queues")
		self.queue: asyncio.Queue = asyncio.Queue()


	def set_bot_configration(self) -> dict:
		os.makedirs(self.cache_path, exist_ok=True)
		handle = open(os.environ.get("CONFIG_PATH"), 'r')
		content = handle.read()
		handle.close()
		bot_data = json.loads(content)
		return {item['name']: item['personality'] for item in bot_data}

	async def construct_context_string(self, comment: Comment):
		things = []
		current_comment = comment

		counter = 0
		try:
			while not isinstance(current_comment, asyncpraw.models.Submission):
				thing = {
					"text": "",
					"counter": 0
				}
				await current_comment.load()
				thing['counter'] = counter
				thing['text'] = current_comment.body
				things.append(current_comment.body)
				counter += 1
				current_comment = await current_comment.parent()
		except asyncio.exceptions.CancelledError:
			logger.info("Task was cancelled, exiting.")
		except Exception as e:
			logger.error(f"Error: {e}")

		things.reverse()
		out = ""
		for i, r in enumerate(things):
			out += f"<|context_level|>{i}<|comment|>{r}"

		out += f"<|context_level|>{len(things) + 1}<|comment|>"
		return out

	def create_post_string(self):
		chosen_bot_key = random.choice(list(self.bot_map.keys()))
		bot_config = self.bot_map[chosen_bot_key]
		constructed_string = f"<|startoftext|><|subreddit|>r/{bot_config}"
		result = self.model_runner.run(constructed_string)

		pattern = re.compile(r'<\|([a-zA-Z0-9_]+)\|>(.*?)(?=<\|[a-zA-Z0-9_]+\|>|$)', re.DOTALL)
		matches = pattern.findall(result)
		result_dict = {key: value for key, value in matches}
		self.queue.put_nowait({
			'title': result_dict.get('title'),
			'text': result_dict.get('text'),
			'image_path': None,
			'bot': chosen_bot_key,
			'subreddit': os.environ.get("SUBREDDIT_TO_MONITOR")
		})

	async def check_queue_hourly(self):
		while True:
			try:
				await asyncio.sleep(3600)
				if not self.queue.empty():
					logger.info("Queue Message Present, processing")
					result = self.queue.get_nowait()
					await self.create_reddit_post(result)
				else:
					logger.debug("Queue is empty.")
					self.create_post_string()
			except Exception as e:
				logger.error(e)
				continue

	async def create_reddit_post(self, data: dict):
		bot = data.get("bot")
		subreddit_name = data.get("subreddit")
		new_reddit = asyncpraw.Reddit(site_name=bot)
		try:
			subreddit = await new_reddit.subreddit(subreddit_name)
			title = data.get("title")
			text = data.get('text')
			result = await subreddit.submit(title, selftext=text)
			await result.load()
			logger.info(f"{bot} has Created A Submission: at https://www.reddit.com{result.permalink}")
		except Exception as e:
			logger.error(e)
		finally:
			await new_reddit.close()


	async def handle_new_submissions(self, subreddit):
		db = shelve.open(str(self.cache_path))
		new_reddit = None
		try:
			await subreddit.load()
			async for submission in subreddit.new(limit=5):
				await submission.load()
				if 'imgur.com' in submission.url or 'i.redd.it' in submission.url:
					logger.info(f":: Submission contains image URL: {submission.url}")
					text = await self.caption_processor.caption_image_from_url(submission.url)
				else:
					logger.debug(f":: Submission does not contain image URL: {submission.url}")
					text = submission.selftext

				for bot in self.bot_map.keys():
					personality = self.bot_map[bot]
					mapped_submission = {
						"subreddit": 'r/' + personality,
						"title": submission.title,
						"text": text
					}
					db[submission.id] = mapped_submission
					mapped_submission = db.get(submission.id)

					constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}<|context_level|>0<|comment|>"
					bot_reply_key = f"{bot}_{submission.id}"
					if bot_reply_key in db:
						continue

					try:
						new_reddit = asyncpraw.Reddit(site_name=bot)
						result = self.model_runner.run(constructed_string)
						submission = await new_reddit.submission(submission.id)
						reply_text = ModelRunner.split_token_first_comment(prompt=constructed_string, completion=result)
						if reply_text is None:
							logger.error(":: Failed to split first comment text")
						else:
							reply = await submission.reply(reply_text)
							await reply.load()
							logger.info(f"{bot} has Replied to Submission: at https://www.reddit.com{reply.permalink}")

						db[bot_reply_key] = True
					except Exception as e:
						logger.error(f"Failed to reply, {e}")
						db[bot_reply_key] = True
					finally:
						if new_reddit is None:
							pass
						else:
							await new_reddit.close()
		finally:
			db.close()

	async def send_comment_reply(self, comment: Comment, responding_bot: str, reply_text: str):
		new_reddit = asyncpraw.Reddit(site_name=responding_bot)
		try:
			await comment.load()
			comment = await new_reddit.comment(comment.id)
			reply = await comment.reply(reply_text)
			await new_reddit.close()
			logger.info(
				f"{responding_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")
		except Exception as e:
			logger.error(e)
		finally:
			await new_reddit.close()

	async def handle_new_comments(self, subreddit, reddit):
		with shelve.open(str(self.cache_path)) as db:
			async for comment in subreddit.stream.comments(skip_existing=False, pause_after=0):
				await self.handle_new_submissions(subreddit)
				if comment.id in db:
					await self.handle_new_submissions(subreddit)
					continue
				if comment is None:
					await self.handle_new_submissions(subreddit)
					continue

				submission_id = comment.submission

				bots = list(self.bot_map.keys())
				responding_bot = random.choice(bots)
				personality = self.bot_map[responding_bot]

				submission = await reddit.submission(submission_id)
				mapped_submission = {
					"subreddit": 'r' + '/' + personality,
					"title": submission.title,
					"text": submission.selftext
				}

				constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
				constructed_string += await self.construct_context_string(comment)
				result = self.model_runner.run(constructed_string)
				cleaned_text: str = ModelRunner.clean_text(result, constructed_string)
				await self.send_comment_reply(comment, responding_bot, cleaned_text)
				db[comment.id] = True

	async def run(self):
		self.create_post_string()
		task = asyncio.create_task(self.check_queue_hourly())
		while True:
			try:
				reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
				sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
				subreddit = await reddit.subreddit(sub_names)
				await self.handle_new_comments(subreddit=subreddit, reddit=reddit)
			except Exception as e:
				logger.error(f"An error has occurring during the primary loop, continuing {e}")
				continue
