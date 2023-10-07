import base64
import hashlib
import io
import json
import random
import re
import time

from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
import numpy as np
import pandas
import logging

import praw
import requests
import os
import shelve
from PIL import Image
from dotenv import load_dotenv
from praw.models import Submission
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
logging.getLogger("azure").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AzureCloudStorage:
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

	def write_output_to_table_storage_row(self, subject_key, iteration, title, caption, style, final_remote_path):
		import datetime
		inverted_ticks = str(int((datetime.datetime.max - datetime.datetime.utcnow()).total_seconds()))
		entity = {
			"PartitionKey": inverted_ticks + "-" + subject_key,
			"RowKey":  inverted_ticks + "-" + subject_key + "-" + str(iteration),
			"TimeStamp": datetime.datetime.now(),
			"prompt": caption,
			"title": title,
			"style": style,
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


class RedditPoster:
	def __init__(self):
		self.reddit = praw.Reddit(site_name=os.environ.get('IMAGE_BOT_ACCOUNT'))
		self.flair_map = json.loads(open('data/flair-map.json', 'r').read())
		self.flair_map = {k.lower(): v for k, v in self.flair_map.items()}
		self.last_submission_id = None

	def create_submission(self, model_name: str, title: str, images: dict, info_string: str):
		try:
			target_sub = "CoopAndPabloArtHouse"

			sub = self.reddit.subreddit(target_sub)

			latest_submission = sub.new(limit=1)

			latest_submission = latest_submission.__next__()

			self.last_submission_id = latest_submission.id

			flair_id = self.flair_map.get(model_name.lower())

			sub.submit_gallery(title=f"{title}", images=images, flair_id=flair_id)

			while True:
				submission: Submission = sub.new(limit=1).__next__()
				if submission.id == self.last_submission_id:
					time.sleep(5)
					continue
				else:
					self.last_submission_id = submission.id
					break
			submission.mod.approve()
			submission.reply(body=info_string[0:9999])

		except Exception as e:
			logger.exception(e)


class ImageBot:
	def __init__(self):
		self.loras = None
		self.loras = self.get_loras()
		self.azure_cloud_storage = AzureCloudStorage()

	def get_image(self,
				  caption: str,
				  negative_prompt: str,
				  steps: int,
				  scale: int,
				  denoising_strength: float,
				  sampler_name: str,
				  upscaler: str,
				  lora_name: str,
				  info_string: str,
				  title: str,
				  subject: str,
				  lora: str):

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
			"batch_size": 8,
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
			_response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img",
									  headers={'accept': 'application/json', 'Content-Type': 'application/json'},
									  json=data)
			r = _response.json()
			out_path = f"D:\\code\\repos\\reddit-bot-gpt2-xl\\output\\{lora_name}\\"
			os.makedirs(out_path, exist_ok=True)
			image_hash = None
			data = []
			for i, _ in enumerate(r['images']):
				image = Image.open(io.BytesIO(base64.b64decode(_.split(",", 1)[0])))
				image_hash = hashlib.md5(image.tobytes()).hexdigest()
				save_path = os.path.join(out_path, f'{image_hash}-{i}.png')
				image.save(save_path)
				data.append({
					'image_path': save_path,
					'caption': caption,
				})
				self.azure_cloud_storage.write_image_to_cloud(save_path)
				self.azure_cloud_storage.write_output_to_table_storage_row(
					subject_key=lora_name,
					iteration=i,
					title=title,
					caption=caption,
					style=lora,
					final_remote_path=f"https://ajdevreddit.blob.core.windows.net/images/{os.path.basename(save_path)}"
				)
			with open(os.path.join(out_path, f'{image_hash}.txt'), 'w', encoding='utf-8') as f:
				f.write(info_string)
			return data
		except Exception as e:
			logger.exception(e)
			return {}

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


class DataMapper:
	def __init__(self):
		self.caption_lookup: dict = json.loads(open('data/caption-lookup.json', 'r').read())
		self.lora_api_response: dict = json.loads(open('data/lora-api-response.json', 'r').read())
		self.model_type_negatives: dict = pandas.read_csv('data/model-type-negatives.tsv', sep='\t').to_dict(
			orient='records')
		self.negative_prompts: dict = json.loads(open('data/negative-prompts.json', 'r').read())
		self.black_list_loras: list = open('data/blacklist.txt', 'r').read().split(',')


class UtilityFunctions:
	def __init__(self):
		self.data_mapper = DataMapper()
		self.image_bot = ImageBot()
		self.reddit_poster = RedditPoster()

	def mask_social_text(self, text):
		masked_text = re.sub(r'#\w+', '', text)
		masked_text = re.sub(r'@\w+', '', masked_text)
		masked_text.strip()
		return masked_text

	def get_lora_object(self, static_lora) -> dict:
		lora_name = static_lora['alias']
		lora_model = static_lora['name']
		all_lora_data = self.data_mapper.caption_lookup[lora_name]
		for i, random_lora_data in enumerate(all_lora_data):
			title = random_lora_data.get('title')
			caption = random_lora_data.get('caption')

			data_dict = list(static_lora['metadata']['ss_tag_frequency'].values())[0]
			sorted_data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[1], reverse=True)}

			all_known_captions = list(set(sorted_data_dict.keys()))

			tfidf_vectorizer = TfidfVectorizer()
			tfidf_vectorizer.fit(all_known_captions + [caption])
			input_transform_vector = tfidf_vectorizer.transform([caption])
			all_known_captions_vector = tfidf_vectorizer.transform(all_known_captions)

			similarity_scores = cosine_similarity(input_transform_vector, all_known_captions_vector)

			most_similar_idx = np.argmax(similarity_scores)

			best_caption = all_known_captions[most_similar_idx]
			random_lora_type = \
			[item['type'] for item in self.data_mapper.model_type_negatives if item['name'] == lora_name][0]
			negative_prompt = self.data_mapper.negative_prompts.get(random_lora_type)

			negative_prompt_string = ", ".join(negative_prompt).strip()

			best_caption += f" <lora:{lora_model}:1>"
			result = {
				'title': title,
				'prompt': best_caption,
				'subject': lora_name,
				'negative_prompt': negative_prompt_string.strip(),
				'lora': lora_model,
				'stash-name': f"{lora_model}-{i}"
			}
			yield result

	def run_generation(self, lora_prompt):
		denoising_strength = round(random.uniform(0.05, 0.2), 2)
		num_steps = random.randint(20, 20)
		cfg_scale = random.randint(7, 7)
		sampler_name = "DDIM"
		upscaler = "Lanczos"
		try:
			title = self.mask_social_text(lora_prompt['title'])
			subject = lora_prompt['subject']
			base_caption = lora_prompt['prompt']
			lora = lora_prompt['lora']
			negative_prompt = lora_prompt['negative_prompt']
			if subject in self.data_mapper.black_list_loras:
				return None
		except Exception as e:
			logger.exception(e)
			return None

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
		try:
			images = self.image_bot.get_image(
				caption=base_caption,
				negative_prompt=negative_prompt,
				steps=num_steps,
				scale=cfg_scale,
				denoising_strength=denoising_strength,
				sampler_name="DDIM",
				upscaler=upscaler,
				lora_name=lora,
				info_string=info_string,
				title=title,
				subject=subject,
				lora=lora)
			if len(images) == 0:
				return None
			self.reddit_poster.create_submission(
				model_name=lora_prompt['subject'],
				title=title,
				images=images,
				info_string=info_string
			)
			return None
		except Exception as e:
			logger.exception(e)
			return None


if __name__ == '__main__':
	utility_functions: UtilityFunctions = UtilityFunctions()
	shelve_path = 'bruh.shelve'
	with shelve.open(shelve_path) as db:
		responses = utility_functions.data_mapper.lora_api_response
		for elem in tqdm(responses, desc=f"Generating Images for All Responses", total=len(responses)):
			lora_things = utility_functions.get_lora_object(elem)
			lora_things_list = list(lora_things)
			for lora_thing in tqdm(lora_things_list, desc=f"Generating Images for {elem.get('name')}", total=len(lora_things_list)):
				stash_name = lora_thing.get('stash-name')
				prompt = lora_thing.get('prompt')
				if lora_thing.get('stash-name') in db:
					tqdm.write(f"\n:: Skipping {lora_thing.get('stash-name')}")
					continue
				if lora_thing is None:
					tqdm.write(f"\n:: Skipping lora thing as it is None")
					continue
				else:
					try:
						tqdm.write(f"\nGenerating Images for {stash_name} - {prompt}")
						utility_functions.run_generation(lora_thing)
						tqdm.write(f"\n:: Finished Generating Images for {stash_name} - {prompt}")
						db[lora_thing.get('stash-name')] = True
					except Exception as e:
						logger.exception(e)
						continue
