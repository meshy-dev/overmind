# -*- coding: utf-8 -*-

# -- stdlib --
import logging
import os
import sys

# -- third party --

# -- own --
from overmind.utils.escapes import escape_codes


# -- code --
class MyFormatter(logging.Formatter):
    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color
        self.color_mapping = {
            'CRITICAL': 'bold_red',
            'ERROR': 'red',
            'WARNING': 'yellow',
            'INFO': 'green',
            'DEBUG': 'blue',
        }

    def format(self, rec):

        if rec.exc_info:
            s = []
            s.append('>>>>>>' + '-' * 74)
            s.append(self._format(rec))
            import traceback
            s.append(''.join(traceback.format_exception(*rec.exc_info)).strip())
            s.append('<<<<<<' + '-' * 74)
            return '\n'.join(s)
        else:
            return self._format(rec)

    def _format(self, rec):
        import time
        rec.message = rec.getMessage()
        lvl = rec.levelname
        prefix = '[{} {} {} #{} {}:{}]'.format(
            lvl[0],
            time.strftime('%y%m%d %H:%M:%S'),
            rec.name,
            os.getpid(),
            rec.module,
            rec.lineno,
        )
        if self.use_color:
            E = escape_codes
            M = self.color_mapping
            prefix = f"{E[M[lvl]]}{prefix}{E['reset']}"

        return f'{prefix} {rec.message}'


def _get_log_level(default_level: int) -> int:
    override_level = os.environ.get("OVERMIND_LOG_LEVEL", "")
    if not override_level:
        return default_level
    # logging.getLevelNamesMapping() is only available since py 3.11,
    # but we are using py 3.10 in production.
    level_map = {
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARNING,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    return level_map.get(override_level.upper(), logging.DEBUG)


def init(default_level, logfile):
    root = logging.getLogger()
    root.setLevel(_get_log_level(default_level))

    # https://no-color.org/
    # Honor the NO_COLOR env var
    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR", "")
    fmter = MyFormatter(use_color=use_color)
    std = logging.StreamHandler(stream=sys.stdout)
    std.setFormatter(fmter)
    root.addHandler(std)

    logging.getLogger('sentry.errors').setLevel(1000)

    if logfile:
        from logging.handlers import WatchedFileHandler
        filehdlr = WatchedFileHandler(logfile)
        filehdlr.setFormatter(fmter)
        root.addHandler(filehdlr)
