from enum import Enum
from functools import total_ordering

from Utils import File, Time
from IPython.display import HTML, display
import tabulate


class LogOutput(Enum):
    CONSOLE = 0
    FILE = 1
    BOTH = 2


@total_ordering
class LogLevel(Enum):
    DEBUG = 0
    FINE = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

    def __lt__(self, other: 'LogLevel'):
        return self.value < other.value


class Log(object):

    """
    Log provides methods to log information to console and/or file.
    """

    output: LogOutput = LogOutput.CONSOLE
    path: str
    filename: str = "logfile.log"
    level: LogLevel = LogLevel.INFO

    @staticmethod
    def tabulate(data: list, **kwargs):
        display(HTML(tabulate.tabulate(data, headers='firstrow', tablefmt='html', **kwargs)))

    @staticmethod
    def debug(message: str, timestamp=True, newline=True, header=False):
        if Log.level > LogLevel.DEBUG:
            return

        Log.to_output(
            message=message,
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def fine(message: str, timestamp=True, newline=True, header=False):
        if Log.level > LogLevel.FINE:
            return

        Log.to_output(
            message=message,
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def info(message: str, timestamp=True, newline=True, header=False):
        if Log.level > LogLevel.INFO:
            return

        Log.to_output(
            message=message,
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def warning(message: str, timestamp=True, newline=True, header=False):
        if Log.level > LogLevel.WARNING:
            return

        Log.to_output(
            message=f"WARNING: {message}",
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def get_log_file():
        return f"{Log.path}/{Log.filename}"

    @staticmethod
    def to_output(message: str, timestamp=True, newline=True, header=False):
        message = create_string(message, timestamp, header)

        if Log.output == LogOutput.CONSOLE or Log.output == LogOutput.BOTH:
            print(message, end=get_newline(newline))

        if Log.output == LogOutput.FILE or Log.output == LogOutput.BOTH:
            message = message + get_newline(newline)
            File.write_file(Log.get_log_file(), message, "a")

    @staticmethod
    def create_directory():
        File.create_directory(Log.path)

    @staticmethod
    def init(path: str, **kwargs):
        Log.level = kwargs.pop("level", LogLevel.INFO)
        Log.output = kwargs.pop("output", LogOutput.CONSOLE)
        Log.path = path
        Log.clear()
        Log.create_directory()

        Log.info("Logger initialized.")
        Log.fine(f"Log level: {Log.level.name}.")
        Log.fine(f"Log path: {Log.path}.")

    @staticmethod
    def clear():
        if File.file_exists(Log.get_log_file()):
            File.delete_file(Log.get_log_file())


def create_string(content, timestamp=True, header=False):
    string = "" if not header else "\n"
    if timestamp:
        string += f"[{Time.get_timestamp()}] "
    string += content

    return string


def get_newline(newline) -> str:
    return '\n' if newline is True else ''
