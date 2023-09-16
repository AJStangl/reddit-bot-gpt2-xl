from core.bot.bot import RedditRunner


if __name__ == '__main__':
	import asyncio
	asyncio.run(RedditRunner().run())
