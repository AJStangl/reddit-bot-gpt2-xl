import gc
import json
import logging
import os
from io import BytesIO
from typing import Optional
import random
import aiohttp
import asyncpraw
import asyncprawcore
import torch
from accelerate import Accelerator
from asyncpraw.models import Comment, Subreddit, Submission
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlipProcessor, BlipForConditionalGeneration, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import re
from transformers import logging as transformers_logging
from transformers import pipeline
import asyncprawcore


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
		scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(model_base_path, subfolder="scheduler")
		safety_checker: StableDiffusionSafetyChecker = None
		feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(model_base_path, subfolder='feature_extractor')
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

	@torch.autocast("cuda")
	def create_image(self, prompt: str) -> (str, dict): # make async
		pipe: StableDiffusionPipeline = self.assemble_model(self.model_path)
		negative_prompt = "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears"
		default_prompt = "realistic, high quality, hd"

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

		finally: # not sure if good idea of makes things worse
			del pipe
			torch.cuda.empty_cache()


class ModelRunner:
	def __init__(self, model_path: str):
		self.model_path: str = model_path
		self.device: torch.device = torch.device('cuda')
		self.tokenizer: GPT2Tokenizer = self.load_tokenizer(self.model_path)
		self.text_model: GPT2LMHeadModel = self.load_model(self.model_path)
		self.detoxify: pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=self.device)
		self.caption_processor: CaptionProcessor = CaptionProcessor()
		self.image_generator: ImageGenerator = ImageGenerator(model_path=os.environ.get("IMAGE_MODEL_PATH"), device_name="cuda")
		self.text_model.to(self.device)

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
			raise Exception("I blew the fuck up exception",e)
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

	def load_tokenizer(self, model_path: str) -> GPT2Tokenizer:
		tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_path)
		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer

	def load_model(self, model_path: str) -> GPT2LMHeadModel:
		model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)
		return model

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
		self.cache_path: str = os.path.join("cache", "cache")
		self.bot_map: dict = self.set_bot_configration()
		logger.info(":: Initializing Model")
		self.model_runner: ModelRunner = ModelRunner(os.environ.get("MODEL_PATH"))
		self.queue: asyncio.Queue = asyncio.Queue()


	def set_bot_configration(self) -> dict:
		os.makedirs(self.cache_path, exist_ok=True)
		with open(os.environ.get("CONFIG_PATH"), 'r') as handle:
			return {item['name']: item['personality'] for item in json.loads(handle.read())}

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
		constructed_string = f"<|startoftext|><|subreddit|>r/{bot_config}" # make configurable
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

	async def create_post_hourly_task(self):
		while True:
			try:
				await asyncio.sleep(3600) # 1 hour, make it a configuration
				if self.queue.empty():
					logger.debug("Queue is empty. Creating new post item")
					self.create_post_string()
					continue
				else:
					await asyncio.sleep(30) # need to task somewhere...
					continue
			except Exception as e:
				logger.error(e)
				continue

	async def check_post_queue_async(self):
		try:
			if not self.queue.empty(): # process post
				logger.info("Queue Message Present, processing")
				result = self.queue.get_nowait() # maybe make this a blocking call?
				await self.create_reddit_post(result)
			else:
				logger.debug("Queue is empty.")
		except Exception as e:
			logger.error(e)

	async def create_reddit_post(self, data: dict):
		bot = data.get("bot")
		subreddit_name = data.get("subreddit")
		new_reddit = asyncpraw.Reddit(site_name=bot)
		create_image = random.choice([False, False])
		try:
			subreddit = await new_reddit.subreddit(subreddit_name)
			await subreddit.load() # race... condition ... here
			title = data.get("title")
			text = data.get('text')
			if create_image: # THIS WILL FUCKING BLOW UP ALWAYS, IDK WHY
				image_path = self.model_runner.image_generator.create_image(data.get("text"))
				submission: Submission = await subreddit.submit_image(title, image_path, without_websockets=True, nsfw=True)
				logger.info(f"{bot} has Created A Submission With Image: at https://www.reddit.com")
				return
			else:
				result = await subreddit.submit(title, selftext=text) # text post
				await result.load()
				logger.info(f"{bot} has Created A Submission: at https://www.reddit.com{result.permalink}")
		except Exception as e:
			logger.error(e)
			raise e
		finally:
			await new_reddit.close()

	async def handle_submission_stream(self, subreddit: Subreddit):
		with shelve.open(str(self.cache_path)) as db:
			new_reddit = None
			try:
				await subreddit.load()
				async for submission in subreddit.new(limit=5):
					await submission.load() # race condition here, need to fix
					if 'imgur.com' not in submission.url and 'i.redd.it' not in submission.url:
						logger.debug(f":: Submission does not contain image URL: {submission.url}")
						text = submission.selftext
					else:  # hack, hack, hack, think about galery batch captioning
						logger.info(f":: Submission contains image URL: {submission.url}")
						text_key = f"{submission.id}-text"
						if db.get(text_key) is not None:
							text = db.get(text_key)
						else:
							text = await self.model_runner.caption_processor.caption_image_from_url(submission.url)
							db[text_key] = text

					for bot in self.bot_map.keys():
						personality = self.bot_map[bot]
						mapped_submission = {
							"subreddit": 'r/' + personality, # something, something, configuration
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
							reply_text = self.model_runner.split_token_first_comment(prompt=constructed_string, completion=result)
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

	async def handle_comment_stream(self, comment: Comment, responding_bot: str, reply_text: str):
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

	async def primary_process(self, subreddit, reddit):
		counter = 0
		with shelve.open(str(self.cache_path)) as db:
			async for comment in subreddit.stream.comments(skip_existing=True, pause_after=0):
				await self.check_post_queue_async()
				counter += 1
				if counter % 10 == 0:
					await self.handle_submission_stream(subreddit)
				if comment is None:
					await asyncio.sleep(1)
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

				if int(submission.num_comments) > 250: # should probably make this a configuration...
					logger.debug(f":: Comment Has More Than 250 Replies, Skipping")
					db[comment.id] = True
					continue

				constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
				constructed_string += await self.construct_context_string(comment) # continue building the input, cache as we go along. (Really need to re-think caching mechanism)
				result = self.model_runner.run(constructed_string)
				if result is None:
					db[comment.id] = True
					continue
				else:
					cleaned_text: str = self.model_runner.clean_text(result, constructed_string)
					await self.handle_comment_stream(comment, responding_bot, cleaned_text)
					db[comment.id] = True

	async def run(self):
		asyncio.create_task(self.create_post_hourly_task())
		while True:
			try:
				reddit = asyncpraw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
				sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
				subreddit = await reddit.subreddit(sub_names)
				await self.primary_process(subreddit=subreddit, reddit=reddit)
			except asyncprawcore.exceptions.RequestException as e:
				logger.error(e)
				continue
			except Exception as e:
				logger.error(f"An error has occurring during the primary loop, continuing {e}")
				raise e
