import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurationManager:
	def __init__(self):
		self.bot_map: dict = self.read_bot_configuration()
		self.personality_list = self.read_topics_file()

	def read_bot_configuration(self) -> dict:
		try:
			with open(os.environ.get("CONFIG_PATH"), 'r') as f:
				return {item['name']: item['personality'] for item in json.load(f)}
		except Exception as e:
			logger.exception(e)
			raise e

	def read_topics_file(self):
		try:
			with open(os.environ.get("TOPICS_PATH"), 'r') as f:
				content = f.read()
				lines = content.split('\n')
				return [item for item in lines if item != ""]
		except Exception as e:
			logger.exception(e)
			raise e
