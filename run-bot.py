import logging
import signal
import time
import warnings
import os

from dotenv import load_dotenv

from core.components.text.services.file_queue_caching import FileCacheQueue

warnings.filterwarnings("ignore")

load_dotenv()

from core.components.text.reddit_run_bot import Bot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
	logger.info("Caught signal, stopping threads.")
	exit(1)



if __name__ == '__main__':
	file_stash: FileCacheQueue = FileCacheQueue(os.environ.get("CACHE_PATH"))
	main_bot_thread: Bot = Bot(name='reddit-bot', file_stash=file_stash)
	main_bot_thread.start()

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	try:
		time.sleep(.01)
	except KeyboardInterrupt:
		logger.info(":: Caught KeyboardInterrupt, stopping threads.")
		main_bot_thread._stop_event.set()
		time.sleep(1)