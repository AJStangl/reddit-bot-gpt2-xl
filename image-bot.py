import base64
import hashlib
import io
import threading
import os
from dataclasses import dataclass

import praw
import json

from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient
from praw.models import Submission
import time
from dotenv import load_dotenv
import logging
import requests
import random
from PIL import Image

load_dotenv()
logging.getLogger("azure").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class LoraData(object):
	def __init__(self, name: str, alias: str, path: str):
		self.name = name
		self.alias = alias
		self.path = path
		self.title_data: list = None


class LoraGenerator:
	def __init__(self):
		self.subjects: list = None
		self.title_caption_dict: dict = None



class ImageGenerator:
	def __init__(self):
		self.url: str = "http://127.0.0.1:7860/sdapi/v1/txt2img"
		self.headers: dict = {'accept': 'application/json', 'Content-Type': 'application/json'}
		self.out_path: str = f"D:\\code\\repos\\reddit-bot-gpt2-xl\\output\\"
		self.denoising_strength: float = 0.0,
		self.upscaler: str = "Lanczos"
		self.caption: str = ""
		self.negative_prompt: str = ""
		self.steps: int = 0,
		self.scale: int = 0,
		self.sampler_name = "DDIM"
		self.image_generation_parameters: dict = {
			"enable_hr": True,
			"denoising_strength": self.denoising_strength,
			"firstphase_width": 0,
			"firstphase_height": 0,
			"hr_scale": 2,
			"hr_upscaler": self.upscaler,
			"hr_second_pass_steps": 20,
			"hr_resize_x": 1024,
			"hr_resize_y": 1024,
			"hr_sampler_name": "",
			"hr_prompt": f"{self.caption}",
			"hr_negative_prompt": f"{self.negative_prompt}",
			"prompt": f"{self.caption}",
			"styles": [""],
			"seed": -1,
			"subseed": -1,
			"subseed_strength": 0,
			"seed_resize_from_h": -1,
			"seed_resize_from_w": -1,
			"sampler_name": self.sampler_name,
			"batch_size": 4,
			"n_iter": 1,
			"steps": self.steps,
			"cfg_scale": self.scale,
			"width": 512,
			"height": 512,
			"restore_faces": True,
			"tiling": False,
			"do_not_save_samples": False,
			"do_not_save_grid": False,
			"negative_prompt": f"{self.negative_prompt}",
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

	def get_image_out_path(self, lora_name: str) -> str:
		return os.path.join(self.out_path, lora_name)

	def set_generation_parameters(self, caption: str):
		self.caption: str = caption
		self.denoising_strength = round(random.uniform(0.05, 0.2), 2)
		self.steps: int = random.randint(20, 20)
		self.scale: int = random.randint(7, 7)

	def generate_image(self, caption: str, lora_name: str) -> str:
		self.set_generation_parameters(caption)

		response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", headers={'accept': 'application/json', 'Content-Type': 'application/json'}, json=self.image_generation_parameters)
		if response.status_code == 200:
			response = response.json()
			out_path = self.get_image_out_path(lora_name)
			os.makedirs(out_path, exist_ok=True)
			image_hash = None
			data = []
			for i, _ in enumerate(response['images']):
				image = Image.open(io.BytesIO(base64.b64decode(_.split(",", 1)[0])))
				image_hash = hashlib.md5(image.tobytes()).hexdigest()
				save_path = os.path.join(out_path, f'{image_hash}-{i}.png')
				image.save(save_path)
				if len(caption) > 180:
					caption = caption[:177] + "..."
				data.append({
					'image_path': save_path,
					'caption': caption,
				})
		return None




class AzureHandler:
	def __init__(self):
		self.service_client: TableServiceClient = TableServiceClient(endpoint=os.environ["AZURE_TABLE_ENDPOINT"], credential=AzureNamedKeyCredential(os.environ["AZURE_ACCOUNT_NAME"], os.environ["AZURE_ACCOUNT_KEY"]))
		self.blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])

	def write_image_to_cloud(self, final_image_path: str):
		blob_client = self.blob_service_client.get_blob_client(container="images", blob=os.path.basename(final_image_path))
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

	def write_output_to_table_storage_row(self, subject_key, iteration, title, caption, style, final_remote_path):
		import datetime
		inverted_ticks = str(int((datetime.datetime.max - datetime.datetime.utcnow()).total_seconds()))
		entity = {
			"PartitionKey": inverted_ticks + "-" + subject_key,
			"RowKey": inverted_ticks + "-" + subject_key + "-" + str(iteration),
			"TimeStamp": datetime.datetime.now(),
			"prompt": caption,
			"title": title,
			"style": style,
			"path": final_remote_path
		}
		table_client: TableClient = self.service_client.get_table_client("generations")
		try:
			table_client.upsert_entity(entity=entity)
		except Exception as e:
			logger.exception(f"{e}")
			table_client.close()



class RedditHandler:
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

			if len(title) == 0:
				title = "Untitled"
			if len(title) > 179:
				title = title[0:179]

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



class ImageBot(threading.Thread):
	def __init__(self, name: str, reddit_handler: RedditHandler, azure_handler: AzureHandler, image_generator: ImageGenerator):
		super().__init__(name=name)
		self._stop_event = threading.Event()
		self.reddit_handler = reddit_handler
		self.azure_handler = azure_handler
		self.image_generator = image_generator

	def stop(self):
		logger.info("Stopping ImageBot...")
		self._stop_event.set()

	def run(self):
		while True:
			try:
				time.sleep(1)
			except Exception as e:
				logger.exception(e)
				continue


if __name__ == '__main__':
	data: list = json.loads(open('data/lora-api-response.json', 'r').read())
	title_captions = json.loads(open('data/captions.json', 'r').read())
	all_lora_data: list = []
	final_thing = {} ## TODO: The thing
	for item in data:
		lora_data: LoraData = LoraData(name=item.get("name"), alias=item.get("alias"), path=item.get("path"))
		for title_caption in title_captions:
			i, name = next(enumerate(title_caption.keys()))
			if name == lora_data.alias:
				title_data = title_caption.get(name)
				for t_data in title_data:
					print(t_data)






	# primary_image_generator: ImageGenerator = ImageGenerator()
	# primary_reddit_handler: RedditHandler = RedditHandler()
	# primary_azure_handler: AzureHandler = AzureHandler()
	#
	# image_bot = ImageBot(name="image-bot", reddit_handler=primary_reddit_handler, azure_handler=primary_azure_handler, image_generator=primary_image_generator)
	#
	# image_bot.start()
	# while True:
	# 	time.sleep(1)
	# 	continue