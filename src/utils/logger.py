import logging
from colorama import Fore, Style
from logging.handlers import RotatingFileHandler
import os


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.WHITE,
        "SUCCESS": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        if not record.exc_info:
            level = record.levelname
            if level in self.COLORS:
                record.msg = (
                    f"{self.COLORS[level]}{record.msg}{Style.RESET_ALL}"
                )
        return super().format(record)


class CustomLogger(logging.Logger):
    def __init__(self, name, output_path, log_file_name="clustering.log"):
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # 添加自定义日志级别
        logging.SUCCESS = 25
        logging.addLevelName(logging.SUCCESS, "SUCCESS")

        # 设置文件处理器
        log_file = os.path.join(output_path, log_file_name)
        os.makedirs(output_path, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, message, args, **kwargs)

    def section(self, title):
        self.info(f"\n{'=' * 80}\n{title:^80}\n{'=' * 80}\n")


# Example usage
def setup_logger(output_path, log_file_name="feature_calibration.log"):
    return CustomLogger("main_logger", output_path, log_file_name)
