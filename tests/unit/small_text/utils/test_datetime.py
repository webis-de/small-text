import datetime
import unittest

from small_text.utils.datetime import format_timedelta


class DatetimeUtilsTest(unittest.TestCase):

    def test_negative_timedelta(self):
        td_negative = datetime.timedelta(minutes=-1)
        with self.assertRaises(ValueError):
            format_timedelta(td_negative)

    def test_format_timedelta(self):
        td1 = datetime.timedelta(hours=9, minutes=0, seconds=59)
        self.assertEqual('09:00:59', format_timedelta(td1))

        td2 = datetime.timedelta(minutes=9, seconds=7)
        self.assertEqual('00:09:07', format_timedelta(td2))
