from logger import logger

class file_logger(logger):

    def log(self, log_level, message):
        if (log_level <= self.__log_level__):
            with open('file_log.txt', 'a') as f:
                f.write("{}: {}\n".format(log_level, message))
