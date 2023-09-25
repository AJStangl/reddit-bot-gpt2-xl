import gc
import logging
import os
import random
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import aiohttp
import asyncpraw
import asyncprawcore
import torch
from PIL import Image
from accelerate import Accelerator
from asyncpraw.models import Comment, Submission
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlipProcessor, BlipForConditionalGeneration, CLIPTextModel, \
	CLIPTokenizer, CLIPImageProcessor
from transformers import logging as transformers_logging
from transformers import pipeline

transformers_logging.set_verbosity(transformers_logging.FATAL)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import asyncio
import json


class CaptionProcessor(object):
	def __init__(self, device_name: str = "cuda"):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.device_name = device_name
		self.device = torch.device(self.device_name)
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(
			self.device)

	async def caption_image_from_url(self, image_url: str) -> str:
		result = ""
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(image_url) as response:
					if response.status != 200:
						return ""

					content = await response.read()

			# Move the synchronous code out of the async block
			image = Image.open(BytesIO(content))
			try:
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


class ImageGenerator(object):
	def __init__(self, model_path: str, device_name: str):
		self.model_path = model_path
		self.device_name = device_name
		self.accelerator = Accelerator()

	def assemble_model(self, model_base_path) -> StableDiffusionPipeline:
		vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_base_path, subfolder='vae')
		text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_base_path, subfolder='text_encoder')
		tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_base_path, subfolder='tokenizer')
		unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(model_base_path, subfolder='unet')
		scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(model_base_path,
																							 subfolder="scheduler")
		safety_checker: StableDiffusionSafetyChecker = None
		feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(model_base_path,
																				   subfolder='feature_extractor')
		requires_safety_checker: bool = False

		pipeline = StableDiffusionPipeline(
			vae=vae,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			scheduler=scheduler,
			feature_extractor=feature_extractor,
			safety_checker=safety_checker,
			requires_safety_checker=requires_safety_checker)
		return pipeline

	def create_image_api(self, base_caption):
		import requests
		import base64
		import io
		data = {
			"enable_hr": True,
			"denoising_strength": 0.7,
			"firstphase_width": 0,
			"firstphase_height": 0,
			"hr_scale": 2,
			"hr_upscaler": "Lanczos",
			"hr_second_pass_steps": 20,
			"hr_resize_x": 1024,
			"hr_resize_y": 1024,
			"hr_sampler_name": "",
			"hr_prompt": "",
			"hr_negative_prompt": "",
			"prompt": f"{base_caption}",
			"styles": [""],
			"seed": 2662256029,
			"subseed": -1,
			"subseed_strength": 0,
			"seed_resize_from_h": -1,
			"seed_resize_from_w": -1,
			"sampler_name": "DPM++ 2S a",
			"batch_size": 1,
			"n_iter": 1,
			"steps": 20,
			"cfg_scale": 7,
			"width": 512,
			"height": 512,
			"restore_faces": True,
			"tiling": False,
			"do_not_save_samples": False,
			"do_not_save_grid": False,
			"negative_prompt": "bad anatomy, ugly, deformed",
			"eta": 0,
			"s_min_uncond": 0,
			"s_churn": 0,
			"s_tmax": 0,
			"s_tmin": 0,
			"s_noise": 1,
			"override_settings": {},
			"override_settings_restore_afterwards": True,
			"script_args": [],
			"sampler_index": "DPM++ 2S a",
			"script_name": "",
			"send_images": True,
			"save_images": False,
			"alwayson_scripts": {}
		}

		_response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img",
							  headers={'accept': 'application/json', 'Content-Type': 'application/json'}, json=data)
		r = _response.json()
		for i in r['images']:
			image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
			image.save('temp.png')
			return 'temp.png'

	@torch.autocast("cuda")
	def create_image(self, prompt: str) -> (str, dict):  # make async
		pipe: StableDiffusionPipeline = self.assemble_model(self.model_path)
		negative_prompt = ""
		default_prompt = ""

		try:
			pipe: StableDiffusionPipeline = pipe.to(torch_device="cuda", torch_dtype=torch.float16)
			new_prompt = f"{prompt}, {default_prompt}"
			guidance_scale = random.randint(6, 12)
			num_inference_steps = random.randint(20, 75)

			height = 512
			width = [512, 512]
			initial_image = pipe(
				prompt=new_prompt,
				negative_prompt=negative_prompt,
				height=height,
				width=random.choice(width),
				guidance_scale=guidance_scale,
				num_inference_steps=num_inference_steps).images[0]

			upload_file = f"D:\\code\\repos\\reddit-bot-gpt2-xl\\temp.png"
			initial_image.save(upload_file)
			return upload_file

		except Exception as e:
			logger.error(e)
			return None

		finally:  # not sure if good idea of makes things worse
			del pipe
			torch.cuda.empty_cache()


class ModelRunner:
	def __init__(self, model_path: str):
		self.model_path: str = model_path
		self.device: torch.device = torch.device('cuda')
		self.tokenizer: Optional[GPT2Tokenizer] = self.load_tokenizer(self.model_path)
		self.text_model: Optional[GPT2LMHeadModel] = self.load_model(self.model_path)
		self.detoxify: Optional[pipeline] = self.load_tox()
		self.caption_processor: Optional[CaptionProcessor] = self.load_caption_processor()
		self.image_generator: Optional[ImageGenerator] = self.load_image_generation()
		self.text_model.to(self.device)

	def load_image_generation(self):
		logger.info(":: Loading Image Generation")
		return ImageGenerator(model_path=os.environ.get("IMAGE_MODEL_PATH"), device_name="cuda")

	def load_tox(self):
		logger.info(":: Loading Detoxify")
		return pipeline("text-classification", model="unitary/toxic-bert", device=self.device)

	def load_caption_processor(self):
		logger.info(":: Loading Caption Processor")
		return CaptionProcessor()

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

	def run(self, text) -> str:
		encoding = self.get_encoding(text)
		if len(encoding) > 512:
			logger.info(f"The encoding output {len(encoding)} > 512, not performing operation.")
			return None
		encoding.to(self.device)
		try:
			return self.run_generation(encoding)
		except Exception as e:
			logger.error(e)
			raise Exception("I blew the fuck up exception", e)

	def get_encoding(self, text):
		encoding = self.tokenizer(text, padding=True, return_tensors='pt', truncation=True)
		return encoding

	@torch.no_grad()
	def run_generation(self, encoding):
		try:
			inputs = encoding['input_ids']
			attention_mask = encoding['attention_mask']
			if inputs.size(0) <= 0 or attention_mask.size(0) <= 0:
				logger.error("Inputs Fail: inputs.size(0) <= 0 or attention_mask.size(0) <= 0")
				return None
			if inputs.dim() != 2 or attention_mask.dim() != 2:
				logger.error("Invalid shape. Expected 2D tensor.")
				return None
			if inputs.shape != attention_mask.shape:
				logger.error("Mismatched shapes between input_ids and attention_mask.")
				return None

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
			for i, _ in enumerate(self.text_model.generate(**args)):
				generated_texts = self.tokenizer.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=True)
				generated_texts = generated_texts.split("<|startoftext|>")
				good_line = ""
				for line in generated_texts:
					good_line = line

				temp = "<|startoftext|>" + good_line
				return temp
		except Exception as e:
			logger.error(e)
			raise Exception("I blew the fuck up exception", e)
		finally:
			torch.clear_autocast_cache()
			gc.collect()

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
					# logging.info(completion)
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


class Bot(object):
	def __init__(self):
		self.model_runner = ModelRunner(model_path=os.environ.get("MODEL_PATH"))
		self.bot_map: dict = self.read_bot_configuration()
		self.next_hour_current_time = None
		self.reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))

	async def get_value_by_key(self, key, filename='cache.json'):
		existing_data = await self.load_dict_from_file(filename)
		return existing_data.get(key, None)

	async def set_value_by_key(self, key, value, filename='cache.json'):
		loop = asyncio.get_event_loop()
		existing_data = await self.load_dict_from_file(filename)
		data = {key: value}
		existing_data.update(data)
		await loop.run_in_executor(None, json.dump, existing_data, open(filename, 'w'))

	async def load_dict_from_file(self, filename='cache.json'):
		loop = asyncio.get_event_loop()
		try:
			return await loop.run_in_executor(None, json.load, open(filename, 'r'))
		except FileNotFoundError:
			return {}

	def read_bot_configuration(self) -> dict:
		bot_map = {}
		with open(os.environ.get("CONFIG_PATH"), 'r') as f:
			config = json.load(f)
			for item in config:
				bot_map[item['name']] = item['personality']
		return bot_map

	async def construct_context_string(self, comment: Comment) -> str:
		things = []
		current_comment = comment

		counter = 0
		try:
			while not isinstance(current_comment, asyncpraw.models.Submission):
				thing = {
					"text": "", "counter": 0
				}
				await current_comment.load()
				thing['counter'] = counter
				thing['text'] = current_comment.body
				things.append(current_comment.body)
				counter += 1
				current_comment = await current_comment.parent()
				await asyncio.sleep(0.1)
		except asyncprawcore.exceptions.RequestException as request_exception:
			logger.exception("Request Error", request_exception)
		except asyncio.exceptions.CancelledError as cancellation_error:
			logger.exception("Task was cancelled, exiting.", cancellation_error)
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
		constructed_string = f"<|startoftext|><|subreddit|>r/{bot_config}"  # make configurable
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

	async def create_reddit_post(self) -> None:
		data = self.create_post_string()
		bot = data.get("bot")
		subreddit_name = data.get("subreddit")
		new_reddit = asyncpraw.Reddit(site_name=bot)
		create_image = random.choice([True, False])
		try:
			subreddit = await new_reddit.subreddit(subreddit_name)
			await subreddit.load()
			title = data.get("title")
			text = data.get('text')
			if create_image:
				image_path = self.model_runner.image_generator.create_image_api(data.get("text"))
				submission: Submission = await subreddit.submit_image(title, image_path, nsfw=False)
				logger.info(f"{bot} has Created A Submission With Image: at https://www.reddit.com")
				return
			else:
				result = await subreddit.submit(title, selftext=text)
				await result.load()
				logger.info(f"{bot} has Created A Submission: at https://www.reddit.com{result.permalink}")
		except Exception as e:
			logger.error(e)
			raise e
		finally:
			await new_reddit.close()

	async def process_submission(self, submission):
		if submission is None:
			await asyncio.sleep(.01)
			return

		if str(submission.url).endswith(('.png', '.jpg', '.jpeg')):
			logger.debug(f":: Submission does not contain image URL: {submission.url}")
			text = await self.model_runner.caption_processor.caption_image_from_url(submission.url)

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
			bot_reply_value = await self.get_value_by_key(bot_reply_key)
			if bot_reply_value:
				continue

			new_reddit = None
			try:
				new_reddit = asyncpraw.Reddit(site_name=bot)
				result = self.model_runner.run(constructed_string)
				submission = await new_reddit.submission(submission.id)
				reply_text = self.model_runner.split_token_first_comment(prompt=constructed_string, completion=result)
				if reply_text is None:
					logger.error(f":: Failed to split first comment text for text: {reply_text}")
				else:
					reply = await submission.reply(reply_text)
					await reply.load()
					logger.info(f":: {bot} has Replied to Submission: at https://www.reddit.com{reply.permalink}")
				await self.set_value_by_key(bot_reply_key, True)
			except Exception as e:
				logger.error(f":: Failed to reply with exception: {e}")
				await self.set_value_by_key(bot_reply_key, True)
			finally:
				if new_reddit is None:
					pass
				else:
					await new_reddit.close()

	async def process_comment(self, comment: Comment):
		if comment is None:
			await asyncio.sleep(0.5)
			return

		submission_id = comment.submission
		bots = list(self.bot_map.keys())
		filtered_bot = [x for x in bots if x.lower() != str(comment.author).lower()]
		responding_bot = random.choice(filtered_bot)
		personality = self.bot_map[responding_bot]
		submission = await self.reddit.submission(submission_id)
		mapped_submission = {
			"subreddit": 'r' + '/' + personality,
			"title": submission.title,
			"text": submission.selftext
		}

		if int(submission.num_comments) > int(os.environ.get('MAX_REPLIES')):
			logger.debug(f":: Comment Has More Than 250 Replies, Skipping")
			await self.set_value_by_key(comment.id, True)
			return

		constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
		constructed_string += await self.construct_context_string(comment)
		result = self.model_runner.run(constructed_string)
		if result is None:
			await self.set_value_by_key(comment.id, True)
			return
		else:
			cleaned_text: str = self.model_runner.clean_text(result, constructed_string)
			await self.reply_to_comment(comment, responding_bot, cleaned_text)
			await self.set_value_by_key(comment.id, True)

	async def reply_to_comment(self, comment: Comment, responding_bot: str, reply_text: str):
		new_reddit = asyncpraw.Reddit(site_name=responding_bot)
		try:
			await comment.load()
			comment = await new_reddit.comment(comment.id)
			reply = await comment.reply(reply_text)
			await new_reddit.close()
			logger.info(f":: {responding_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")
		except Exception as e:
			logger.error(e)
		finally:
			await new_reddit.close()

	async def run(self):
		sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		subreddit = await self.reddit.subreddit(sub_names)
		self.next_hour_current_time = await self.get_value_by_key('next_time_to_post')
		count = 0
		while True:
			try:
				async for item in subreddit.stream.comments(skip_existing=False, pause_after=0):
					if datetime.timestamp(datetime.now()) > float(self.next_hour_current_time):
						await self.set_value_by_key('next_time_to_post', datetime.timestamp(datetime.now() + timedelta(hours=1)))
						await self.create_reddit_post()
					else:
						logger.debug(f":: Next Post In {self.next_hour_current_time - datetime.timestamp(datetime.now())} Seconds")

					if count % 10:
						logger.debug(":: Checking For Submissions")
						async for x in subreddit.new(limit=5):
							if isinstance(x, Submission):
								await x.load()
								await self.process_submission(submission=x)
								await asyncio.sleep(1)
								continue

					if item is None:
						count += 1
						await asyncio.sleep(1)
						continue

					comment_key = f"{item.id}-comment"
					comment_seen = await self.get_value_by_key(comment_key)
					if comment_seen:
						count += 1
						await asyncio.sleep(1)
						continue

					if isinstance(item, Comment):
						await self.process_comment(comment=item)
						await self.set_value_by_key(comment_key, True)
						count += 1
						await asyncio.sleep(1)
						continue
					await asyncio.sleep(1)
			except Exception as e:
				if e.args[0] == "I blew the fuck up exception":
					raise e
				else:
					logger.exception("An error has occurred during the primary loop", e)
					count += 1
					continue
