from core.bot.bot import RedditRunner


if __name__ == '__main__':
	import asyncio
	in_error = False
	asyncio.run(RedditRunner().run())
	while True:
		try:
			if in_error:
				print("Restarting...")
				in_error = False
			asyncio.run(RedditRunner().run())
			continue
		except KeyboardInterrupt:
			exit(0)
		except Exception as e:
			if not in_error:
				print(f"Error: {e}")
				in_error = True
			continue