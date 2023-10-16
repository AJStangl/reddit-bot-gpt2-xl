A
        logger.info("Starting " + self.name)
        self.do_thing()
        logger.info("Exiting " + self.name)

    def main_routine(self):
        while True:
            try:
                if self.do_thing():
                    continue
                else:
                    self.stop()
            except Exception as e:
                logger.exception(e, "An exception has occured in MyThread.main_routine")
                continue

    def stop(self):
        self._stop_event.set()
