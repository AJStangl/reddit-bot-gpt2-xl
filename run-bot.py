from threading import Thread
import signal
from reddit_image_bot import ImageRunner
from reddit_run_bot import BotRunner
import warnings
import logging
from dotenv import load_dotenv
import asyncio

warnings.filterwarnings("ignore")
load_dotenv()


def signal_handler(signum, frame):
	# Perform any necessary cleanup here
	logger.info("Caught signal, stopping threads.")
	exit(0)


def run_in_thread(coroutine):
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	loop.run_until_complete(coroutine)


async def main():
	try:
		image_runner: ImageRunner = ImageRunner()
		bot_runner: BotRunner = BotRunner()

		thread_image_worker = Thread(target=run_in_thread, args=(image_runner.run_async(),), daemon=True,
									 name="image-worker")
		thread_text_worker = Thread(target=run_in_thread, args=(bot_runner.run_async(),), daemon=True,
									name="text-worker")

		thread_image_worker.start()
		thread_text_worker.start()

		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)

		while True:
			await asyncio.sleep(1)  # Sleep indefinitely until interrupted

	except Exception as e:
		logger.error("An error occurred", exc_info=True)
		exit(1)


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	asyncio.run(main())
