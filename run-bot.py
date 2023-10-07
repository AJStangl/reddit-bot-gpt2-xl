import os
import sys
import time
from threading import Thread
import signal
from core.components.image.reddit_image_bot import ImageRunner
from core.components.text.reddit_run_bot import BotRunner
import warnings
import logging
from dotenv import load_dotenv
import argparse

warnings.filterwarnings("ignore")
load_dotenv()


def signal_handler(signum, frame):
	logger.info("Caught signal, stopping threads.")
	exit(1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run Reddit bots.')
	parser.add_argument('--mode', choices=['text', 'image', 'both'], default='text', help='Mode to run the bot in.')
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
	if os.path.exists(os.environ.get("LOCK_PATH")):
		for item in os.listdir(os.environ.get("LOCK_PATH")):
			item_path = os.path.join(os.environ.get("LOCK_PATH"), item)
			os.remove(item_path)

	os.makedirs(os.environ.get("LOCK_PATH"), exist_ok=True)
	logger = logging.getLogger(__name__)

	if args.mode in ['image', 'both']:
		image_runner: ImageRunner = ImageRunner()
		thread_image_worker = Thread(target=image_runner.run, args=(), daemon=True, name="image-worker")
		thread_image_worker.start()

	if args.mode in ['text', 'both']:
		bot_runner: BotRunner = BotRunner()
		thread_text_worker = Thread(target=bot_runner.run, args=(), daemon=True, name="text-worker")
		thread_text_worker.start()

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	while True:
		time.sleep(1)
