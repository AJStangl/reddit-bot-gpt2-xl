import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import os.path
import random

import asyncpraw
import asyncpraw.models
import numpy as np
import pandas
import requests
from PIL import Image
from asyncpraw import Reddit
from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logging.getLogger("azure").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedditImageBot:
	def __init__(self):
		self.counter: int = 0
		self.reddit: Reddit = None
		self.flair_map = json.loads(open('data/flair-map.json', 'r').read())
		self.flair_map = {k.lower(): v for k, v in self.flair_map.items()}
		self.last_submission_id = None

	def get_markdown_comment(self, prompt, sub, guidance, num_steps, attention_type, info_string):
		body = f"""\
| Prompt   | Model Name | Guidance   | Number Of Inference Steps | Attention Type   |
|:--------:|:----------:|:----------:|:------------------------:|:----------------:|
| {prompt} |   {sub}    | {guidance} |           {num_steps}     |  {attention_type}|
"""
		body += "\n\n"
		body += info_string
		return body

	def write_image_to_cloud(self, final_image_path: str):
		blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])

		blob_client = blob_service_client.get_blob_client(container="images", blob=os.path.basename(final_image_path))

		try:
			with open(final_image_path, "rb") as f:
				image_data = f.read()
				blob_client.upload_blob(image_data, overwrite=True)
				final_remote_path = f"https://ajdevreddit.blob.core.windows.net/images/{os.path.basename(final_image_path)}"
				return final_remote_path

		except Exception as e:
			logger.exception(e)
			raise Exception(e)

		finally:
			blob_client.close()
			blob_service_client.close()

	def write_output_to_table_storage_row(self, prompt, title, style, final_remote_path):
		import datetime
		inverted_ticks = str(int((datetime.datetime.max - datetime.datetime.utcnow()).total_seconds()))
		entity = {
			"PartitionKey": inverted_ticks + "-" + str(os.path.basename(final_remote_path).split(".")[0]),
			"RowKey": inverted_ticks + "-" + str(os.path.basename(final_remote_path).split(".")[0]),
			"TimeStamp": datetime.datetime.now(),
			"prompt": prompt,
			"title": title,
			"style": str(style),
			"path": final_remote_path
		}
		service_client: TableServiceClient = TableServiceClient(endpoint=os.environ["AZURE_TABLE_ENDPOINT"],
																credential=AzureNamedKeyCredential(
																	os.environ["AZURE_ACCOUNT_NAME"],
																	os.environ["AZURE_ACCOUNT_KEY"]))
		table_client: TableClient = service_client.get_table_client("generations")
		try:
			table_client.upsert_entity(entity=entity)
		except Exception as e:
			logger.exception(f"{e}")
			table_client.close()
			service_client.close()

	async def create_submission(self,
						  title: str,
						  image_output: str,
						  model_name: str,
						  prompt: str,
						  guidance: str,
						  num_steps: str,
						  attention_type: str,
						  info_string: str):

		self.reddit = Reddit(site_name=os.environ.get("IMAGE_BOT_ACCOUNT"))

		try:
			target_sub = "CoopAndPabloArtHouse"

			sub = await self.reddit.subreddit(target_sub)

			latest_submission = sub.new(limit=1)

			latest_submission = await latest_submission.__anext__()

			self.last_submission_id = latest_submission.id

			final_remote_path = self.write_image_to_cloud(image_output)

			flair_id = self.flair_map.get(model_name)

			await sub.submit_image(title=f"{title}", image_path=image_output, nsfw=True, timeout=60, without_websockets=True, flair_id=flair_id)

			while True:
				submission: asyncpraw.models.Submission = await sub.new(limit=1).__anext__()
				if submission.id == self.last_submission_id:
					await asyncio.sleep(5)
					continue
				else:
					self.last_submission_id = submission.id
					break

			info_string += f"""
\nhttps://www.reddit.com/{submission.id}
==============================================			
"""

			logger.info(info_string)
			submission.mod.approve()

			body = self.get_markdown_comment(prompt, model_name, guidance, num_steps, attention_type, info_string)

			await submission.reply(body=body[0:9999])

			self.counter += 1

			self.write_output_to_table_storage_row(prompt, title, sub, final_remote_path)

			os.remove(image_output)

			return True

		except Exception as e:
			logger.exception(e)
			return False
		finally:
			await self.reddit.close()
			self.reddit = None


class DataMapper:
	def __init__(self):
		self.caption_lookup: dict = json.loads(open('data/caption-lookup.json', 'r').read())
		self.lora_api_response: dict = json.loads(open('data/lora-api-response.json', 'r').read())
		self.model_type_negatives: dict = pandas.read_csv('data/model-type-negatives.tsv', sep='\t').to_dict(orient='records')
		self.negative_prompts: dict = json.loads(open('data/negative-prompts.json', 'r').read())
		self.black_list_loras: list = open('data/blacklist.txt', 'r').read().split(',')

	def mask_social_text(self, text):
		import re
		masked_text = re.sub(r'#\w+', '', text)
		masked_text = re.sub(r'@\w+', '', masked_text)
		masked_text.strip()
		return masked_text

	def get_best_caption(self) -> dict:
		try:
			# Get a random lora without "step" in the name
			random_lora = random.choice([item for item in self.lora_api_response if "step" not in item['name']])
			random_lora_name = random_lora['alias']

			# Get random lora data
			random_lora_data = random.choice(self.caption_lookup[random_lora_name])
			title = random_lora_data.get('title')
			caption = random_lora_data.get('caption')

			# Sort the ss_tag_frequency dictionary
			data_dict = list(random_lora['metadata']['ss_tag_frequency'].values())[0]
			sorted_data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[1], reverse=True)}

			# Extract all known captions
			all_known_captions = list(set(sorted_data_dict.keys()))

			# Initialize, fit, and transform using TFIDF Vectorizer
			tfidf_vectorizer = TfidfVectorizer()
			tfidf_vectorizer.fit(all_known_captions + [caption])
			input_transform_vector = tfidf_vectorizer.transform([caption])
			all_known_captions_vector = tfidf_vectorizer.transform(all_known_captions)

			# Calculate similarity scores
			similarity_scores = cosine_similarity(input_transform_vector, all_known_captions_vector)

			# Get the most similar caption's index
			most_similar_idx = np.argmax(similarity_scores)

			# Fetch the best caption
			best_caption = all_known_captions[most_similar_idx]
			random_lora_type = [item['type'] for item in self.model_type_negatives if item['name'] == random_lora_name][0]
			negative_prompt = self.negative_prompts.get(random_lora_type)
			negative_prompt += self.negative_prompts.get("General")
			negative_prompt += self.negative_prompts.get("Universal")

			negative_prompt_string = ", ".join(negative_prompt).strip()

			best_caption += f" <lora:{random_lora_name}:1>"
			masked_title = self.mask_social_text(title)
			result = {
				'title': masked_title,
				'prompt': best_caption,
				'subject': random_lora_name,
				'negative_prompt': negative_prompt_string.strip(),
				'lora': random_lora_name
			}
			return result

		except:
			return None



class ImageBot:
	def __init__(self):
		self.loras = None
		self.loras = self.get_loras()

	def get_image(self, caption: str, negative_prompt: str, steps: int, scale: int, denoising_strength: float, sampler_name: str, upscaler: str):
		data = {
			"enable_hr": True,
			"denoising_strength": denoising_strength,
			"firstphase_width": 0,
			"firstphase_height": 0,
			"hr_scale": 2,
			"hr_upscaler": upscaler,
			"hr_second_pass_steps": 20,
			"hr_resize_x": 1024,
			"hr_resize_y": 1024,
			"hr_sampler_name": "",
			"hr_prompt": f"{caption}",
			"hr_negative_prompt": f"{negative_prompt}",
			"prompt": f"{caption}",
			"styles": [""],
			"seed": -1,
			"subseed": -1,
			"subseed_strength": 0,
			"seed_resize_from_h": -1,
			"seed_resize_from_w": -1,
			"sampler_name": sampler_name,
			"batch_size": 1,
			"n_iter": 1,
			"steps": steps,
			"cfg_scale": scale,
			"width": 512,
			"height": 512,
			"restore_faces": True,
			"tiling": False,
			"do_not_save_samples": False,
			"do_not_save_grid": False,
			"negative_prompt": f"{negative_prompt}",
			"eta": 0,
			"s_min_uncond": 0,
			"s_churn": 0,
			"s_tmax": 0,
			"s_tmin": 0,
			"s_noise": 1,
			"override_settings": {},
			"override_settings_restore_afterwards": True,
			"script_args": [],
			"sampler_index": "DDIM",
			"script_name": "",
			"send_images": True,
			"save_images": False,
			"alwayson_scripts": {}
	}
		try:
			_response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", headers={'accept': 'application/json', 'Content-Type': 'application/json'}, json=data)
			r = _response.json()
			for i in r['images']:
				image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
				image_hash = hashlib.md5(image.tobytes()).hexdigest()
				image.save(f'output\\{image_hash}.png')
				return f'output\\{image_hash}.png'
		except Exception as e:
			print(e)
			return None

	def get_loras(self):
		if self.loras is None:
			try:
				target_loras = []
				if os.path.exists('data/lora-api-response.json'):
					r_json = json.loads(open('data/lora-api-response.json').read())
				else:
					r = requests.get("http://127.0.0.1:7860/sdapi/v1/loras")
					r_json = r.json()

				for elem in r_json:
					if not elem['path'].endswith('.safetensors'):
						continue
					else:
						target_loras.append(elem)
				self.loras = target_loras
				return self.loras
			except Exception as e:
				logger.exception(e)
				self.get_loras()
		return self.loras


class ImageRunner:
	def __init__(self):
		self.reddit_handler: RedditImageBot = RedditImageBot()
		self.image_bot: ImageBot = ImageBot()
		self.data_mapper: DataMapper = DataMapper()
		self.text_lock_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\locks\\text.lock"
		self.image_lock_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\locks\\sd.lock"

	def create_lock(self):
		try:
			with open(self.image_lock_path, "wb") as handle:
				handle.write(b"")
		except Exception as e:
			logging.error(f"An error occurred while creating temp lock: {e}")

	def clear_lock(self):
		try:
			if os.path.exists(self.image_lock_path):
				os.remove(self.image_lock_path)
			else:
				logging.warning(f"Lock file {self.image_lock_path} does not exist.")
		except Exception as e:
			logging.error(f"An error occurred while deleting text lock: {e}")
		# Optionally, re-raise the exception to signal the failure
		# raise

	async def run_async(self):
		while True:
			try:
				while os.path.exists(self.text_lock_path):
					await asyncio.sleep(.1)
					continue
				logger.info(":: Text lock cleared, generating image...")
				denoising_strength = round(random.uniform(0.05, 0.2), 2)
				num_steps = random.randint(20, 20)
				cfg_scale = random.randint(7, 7)
				sampler_name = "DDIM"
				upscaler = "Lanczos"
				try:
					lora_prompt = self.data_mapper.get_best_caption()
					title = lora_prompt['title']
					subject = lora_prompt['subject']
					base_caption = lora_prompt['prompt']
					lora = lora_prompt['lora']
					negative_prompt = lora_prompt['negative_prompt']
					if subject in self.data_mapper.black_list_loras:
						continue
				except Exception as e:
					logger.exception(e)
					continue

				info_string = f"""
==================================================

## Submission Information:

Title: {title}

Subject: {subject}

## Model Information:

Lora Weight(s): {lora}

## Configuration:

Denoising Strength: {denoising_strength}

num_steps: {num_steps}

cfg_scale: {cfg_scale}

sampler_name: {sampler_name}

upscaler: {upscaler}

## Prompt:

==================================================

{base_caption}

{negative_prompt}

==================================================

				"""
				self.create_lock()
				logger.info(":: Generating Image")
				image_path = self.image_bot.get_image(
					caption=base_caption,
					negative_prompt=negative_prompt,
					steps=num_steps,
					scale=cfg_scale,
					denoising_strength=denoising_strength,
					sampler_name="DDIM",
					upscaler=upscaler)

				self.clear_lock()
				logger.info(":: Image Complete, removing lock...")

				try:
					logger.debug(":: Image Generated, submitting to Reddit...")
					await self.reddit_handler.create_submission(
						title=title,
						image_output=image_path,
						model_name=subject,
						prompt=base_caption,
						guidance=cfg_scale,
						num_steps=num_steps,
						attention_type=lora,
						info_string=info_string)
				except Exception as e:
					logger.exception(e)
					continue
			except Exception as e:
				logger.exception(e)
			finally:
				await asyncio.sleep(10)
