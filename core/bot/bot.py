import json
import logging
import os
import random
import re
from typing import Optional

import asyncpraw
import torch
from asyncpraw.models import Comment
from asyncpraw.models import Comment
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from core.finetune.gather import CaptionProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import asyncio
import shelve


class ModelRunner:
	def __init__(self, model_path):
		self.model_path = model_path
		self.device = torch.device('cuda')
		self.tokenizer, self.model = self.load_model_and_tokenizer(self.model_path)
		self.model.to(self.device)

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
		for i, _ in enumerate(self.model.generate(**args)):
			generated_texts = self.tokenizer.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=True)
			generated_texts = generated_texts.split("<|startoftext|>")
			good_line = ""
			for line in generated_texts:
				good_line = line

			temp = "<|startoftext|>" + good_line
			temp += "<|endoftext|>"
			return temp

	@staticmethod
	def split_token(text) -> Optional[dict]:
		try:
			pattern = r'<\|([a-zA-Z0-9_]+)\|>(.*?)((?=<\|[a-zA-Z0-9_]+\|>)|$)'
			re.compile(pattern)

			matches = re.finditer(pattern, text)

			new_tokens = {
			}
			for i, match in enumerate(matches):
				new_tokens[match.group(1)] = match.group(2).strip()

			return new_tokens
		except Exception as e:
			logger.error(e)
			return None

	@staticmethod
	def split_token_first_comment(text) -> Optional[str]:
		pattern = re.compile(r'<\|context_level\|>0<\|comment\|>(.+?)<', re.MULTILINE)
		logger.info(text)
		matches = re.findall(pattern, text)
		if len(matches) == 1:
			return matches[0]
		else:
			logger.error("Failed to split first comment text")
			return None


	def load_model_and_tokenizer(self, model_path: str) -> (GPT2Tokenizer, GPT2LMHeadModel):
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer, model


class RedditRunner(object):
	def __init__(self):
		self.model_runner: ModelRunner = ModelRunner(os.environ.get("MODEL_PATH"))
		handle = open(os.environ.get("CONFIG_PATH"), 'r')
		content = handle.read()
		handle.close()
		bot_data = json.loads(content)
		self.bot_map = {item['name']: item['personality'] for item in bot_data}
		self.caption_processor: CaptionProcessor = CaptionProcessor()
		self.cache_path = "cache"
		os.makedirs(self.cache_path, exist_ok=True)

	async def clear_cache_hourly(self):
		while True:
			logger.info("Clearing cache")
			with shelve.open(str(self.cache_path)) as db:
				keys_to_remove = []  # List to store keys that need to be removed
				for key in db.keys():
					if key == "post":
						keys_to_remove.append(key)

				# Remove items from cache
				for key in keys_to_remove:
					del db[key]

			# Sleep for one hour (3600 seconds)
			await asyncio.sleep(3600)

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
			out += f"<|context_level|>{i}><|text|>{r}"
		return out

	async def handle_new_submissions(self, subreddit):
		db = shelve.open(str(self.cache_path))
		new_reddit = None
		try:
			async for submission in subreddit.stream.submissions():
				if submission is None:
					break
				await submission.load()
				if 'imgur.com' in submission.url or 'i.redd.it' in submission.url:
					logger.info(f"Submission contains image URL: {submission.url}")
					text = self.caption_processor.caption_image_from_url(submission.url)
				else:
					logger.info(f"Submission does not contain image URL: {submission.url}")
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
						reply_text = ModelRunner.split_token_first_comment(result)
						if reply_text is None:
							logger.error("Failed to split first comment text")
							db[bot_reply_key] = True
							pass

						reply = await submission.reply(reply_text)
						logger.info(f"{bot} has Replied to submission: at https://www.reddit.com{reply.permalink}")
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

	async def send_comment_reply(self, comment: Comment, responding_bot: str, tokens: dict):
		new_reddit = asyncpraw.Reddit(site_name=responding_bot)
		try:
			await comment.load()
			comment = await new_reddit.comment(comment.id)
			reply_text = tokens['text']
			reply = await comment.reply(reply_text)
			await new_reddit.close()
			logger.info(
				f"{responding_bot} has Replied to comment for submission: at https://www.reddit.com{reply.permalink}")
		except Exception as e:
			logger.error("I am here!")
			logger.error(e)
		finally:
			await new_reddit.close()

	async def run(self):
		asyncio.create_task(self.clear_cache_hourly())
		while True:
			try:
				reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
				sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
				subreddit = await reddit.subreddit(sub_names)
				await self.handle_new_submissions(subreddit)
				async for comment in subreddit.stream.comments(skip_existing=True, pause_after=0):
					if comment is None:
						logger.info("No new comments, checking for new submissions")
						continue
					submission_id = comment.submission

					responding_bot = random.choice(self.bot_map.keys())
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
					tokens: dict = ModelRunner.split_token(result)
					await self.send_comment_reply(comment, responding_bot, tokens)
			except Exception as e:
				logger.error("An error has occurring during the primary loop, continuing")
				logger.error(e)
				continue
