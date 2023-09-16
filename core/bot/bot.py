import asyncio
import logging
import os
import random
import shelve
from core.finetune.gather import CaptionProcessor
import asyncpraw
from asyncpraw.models import Comment
import re
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
		except:
			return None

	@staticmethod
	def split_token_first_comment(text) -> Optional[str]:
		try:
			pattern = re.compile(r'<\|context_level\|>0<\|comment\|>(.+?)<')
			logger.info(text)
			matches = re.findall(pattern, text)
			if len(matches) > 0:
				return matches[0]
			else:
				logger.error("Failed to split first comment text")
				return None
		except:
			return None



	def load_model_and_tokenizer(self, model_path: str) -> (GPT2Tokenizer, GPT2LMHeadModel):
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer, model


class RedditRunner:
	caption_processor: CaptionProcessor = CaptionProcessor()
	bots = os.environ.get("REDDIT_BOTS_TO_REPLY").split(",")
	cache_path = "cache"
	os.makedirs(cache_path, exist_ok=True)

	async def clear_cache_hourly(self):
		while True:
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

	async def handle_new_submissions(self, subreddit, model_runner, cache_path=cache_path):
		try:
			with shelve.open(str(cache_path)) as db:
				await subreddit.load()
				async for submission in subreddit.new(limit=10):
					await submission.load()
					has_image = False
					if 'imgur.com' in submission.url or 'i.redd.it' in submission.url:
						logger.info(f"Submission contains image URL: {submission.url}")
						has_image = True



					if submission.id not in db:
						text = submission.selftext
						if has_image:
							text = self.caption_processor.caption_image_from_url(submission.url)

						mapped_submission = {
							"subreddit": 'r/' + str(submission.subreddit),
							"title": submission.title,
							"text": text
						}
						db[submission.id] = mapped_submission
					else:
						mapped_submission = db.get(submission.id)

					constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}<|context_level|>0"
					try:
						for bot in os.environ["REDDIT_BOTS_TO_REPLY"].split(","):
							bot_reply_key = f"{bot}_{submission.id}"
							if bot_reply_key in db:
								continue
							result = model_runner.run(constructed_string)
							tokens: str = ModelRunner.split_token_first_comment(result)
							new_reddit = asyncpraw.Reddit(site_name=bot)
							try:
								submission = await new_reddit.submission(submission.id)
								reply_text = tokens
								if reply_text is None:
									db[bot_reply_key] = True
									continue

								try:
									reply = await submission.reply(reply_text)
								except Exception as e:
									logger.error(f"Failed to reply, {e}")

								db[bot_reply_key] = True
								logger.info(f"{bot} has Replied to submission: at https://www.reddit.com{reply.permalink}")
							except Exception as e:
								logger.exception(e)
								db[bot_reply_key] = True
								logger.error(e)
							await new_reddit.close()
					except Exception as e:
						logger.error(e)
						continue
		finally:
			db.close()

	async def run(self, model_runner: ModelRunner = ModelRunner(os.environ.get("MODEL_PATH"))):
		try:
			task = asyncio.create_task(self.clear_cache_hourly())
			submission_map = {}
			reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
			subreddit = await reddit.subreddit(os.environ.get("SUBREDDIT_TO_MONITOR"))
			# TODO: If there is no key call `Post` in the cache, then we need to call `Post` to create a submission

			async for comment in subreddit.stream.comments(skip_existing=True, pause_after=0):
				if comment is None:
					logger.info("No new comments, checking for new submissions")
					await self.handle_new_submissions(subreddit, model_runner)
					continue
				submission_id = comment.submission
				mapped_submission = submission_map.get(submission_id)
				if mapped_submission is None:
					submission = await reddit.submission(submission_id)
					mapped_submission = {
						"subreddit": 'r' + '/' + str(submission.subreddit),
						"title": submission.title,
						"text": submission.selftext
					}
					submission_map[submission_id] = mapped_submission

				constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
				constructed_string += await self.construct_context_string(comment)
				result = model_runner.run(constructed_string)
				tokens: dict = ModelRunner.split_token(result)
				responding_bot = random.choice(self.bots)
				new_reddit = asyncpraw.Reddit(site_name=responding_bot)
				try:
					comment = await new_reddit.comment(comment.id)
					reply_text = tokens['text']
					reply = await comment.reply(reply_text)
					logger.info(f"{responding_bot} has Replied to comment for submission: at https://www.reddit.com{reply.permalink}")
				except Exception as e:
					logger.error(e)
					await new_reddit.close()
					return await self.run()
				finally:
					await new_reddit.close()
		except Exception as e:
			logger.error(e)
			return await self.run(model_runner)

