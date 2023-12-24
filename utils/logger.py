import logging


def mylogger(name: str, logger_save_path: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    f_handler = logging.FileHandler(logger_save_path, mode='a')

    s_handler = logging.StreamHandler()
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger
