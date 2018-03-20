from logger import logger

class file_logger(logger):

    def __init__(self, log_level):
        logger.__init__(self, log_level)
        self.__filename__ = 'file_log.txt'

    def set_filename(self, name):
        self.__filename__ = name

    def log(self, log_level, message):
        if (log_level <= self.__log_level__):
            with open(self.__filename__, 'a') as f:
                f.write("{}: {}\n".format(log_level, message))
