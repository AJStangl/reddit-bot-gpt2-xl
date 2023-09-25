from core.bot.bot import Bot
import logging


async def main():
    bot = Bot()
    try:
        await bot.run()
    except:
        logger.error("An error occurred", exc_info=True)
        exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    import asyncio
    asyncio.run(main())