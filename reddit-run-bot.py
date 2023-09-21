from core.bot.bot import RedditRunner


if __name__ == '__main__':
    import asyncio
    try:
        asyncio.run(RedditRunner().run())
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


