import asyncio
import logging
import os
import random

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
				# if line.__contains__("r/CoopAndPabloPlayHouse"):
				# 	good_line = line
				# 	break

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


	def load_model_and_tokenizer(self, model_path: str) -> (GPT2Tokenizer, GPT2LMHeadModel):
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer, model


class RedditRunner:
	bots = os.environ.get("REDDIT_BOTS_TO_REPLY").split(",")

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


	async def monitor_comments(self, model_runner: ModelRunner = ModelRunner(os.environ.get("MODEL_PATH"))):
		try:
			submission_map = {}
			reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
			subreddit = await reddit.subreddit(os.environ.get("SUBREDDIT_TO_MONITOR"))

			# Capture comments as they come through
			async for comment in subreddit.stream.comments(skip_existing=True, pause_after=0):
				if comment is None:
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
					await reddit.close()
					return await self.monitor_comments()
				finally:
					await new_reddit.close()
		except Exception as e:
			logger.error(e)
			return await self.monitor_comments()
