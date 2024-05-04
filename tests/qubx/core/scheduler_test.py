
from qubx.core.helpers import BasicScheduler, _parse_schedule_spec


class TestScheduler:

    def test_parsing_formats(self):
        specs = [
            '',
            'bar: -5Sec',           # 5 sec before subsciption bar closing
            'bar.5m: -5Sec',        # 5 sec before 5Min bar closing
            'bar:  5Sec',           # 5 sec after subsciption bar opened
            'bar:  +5Sec',          # 5 sec after subsciption bar opened
            'time: 23:59:00 @ Sat', # at 23:59:00 every Saturday
            'time: 23:59 @ Sat',    # at 23:59:00 every Saturday
            'time: 9:30:10',        # every day at 9:30:10
            '* * * * * 45',         # raw cron 
            'cron: * * * * * 45',   # raw cron witch implicit type
            '23:59:00 @ Sun',       # at 23:59:00 every Sunday
            '9:30:10',              # every day at 9:30:10
            '9:30 @ Mon-Fri',       # mon, ... fri at 9:30
            '9:30:22 @ Mon,Fri',    # mon and fri at 9:30:22
            '9:30 , 15:45 @ Mon-Fri', # mon, ... fri at 9:30 and 15:45
            '9:30, 15:45 @ Mon, Fri', # mon and fri at 9:30 and 15:45
            '-5Sec',                # 5 sec before subsciption bar end
            '5Min -5Sec',           # 5 sec before every 5min interval ends
            '1w -1h',               # 1h before every week ends
            '4month -5hour',        # 5h before the end of every 4 months 
            '1week -3hours -10Mins',# 5h before the end of every 4 months 
            '2h30m -5Sec',          # 5 sec before every 2h and 30min interval ends
            'calendar: 23:00 liquidation'
        ]
        res = [
            {},
            {'type': 'bar', 'spec': '-5Sec', 'seconds': '-5'},
            {'type': 'bar', 'timeframe': '5m', 'spec': '-5Sec', 'seconds': '-5'},
            {'type': 'bar', 'spec': '5Sec', 'seconds': '5'},
            {'type': 'bar', 'spec': '+5Sec', 'seconds': '+5'},
            {'type': 'time', 'spec': '23:59:00 @ Sat', 'time': '23:59:00 ', 'by': 'Sat'},
            {'type': 'time', 'spec': '23:59 @ Sat', 'time': '23:59 ', 'by': 'Sat'},
            {'type': 'time', 'spec': '9:30:10', 'time': '9:30:10'},
            {'spec': '* * * * * 45'},
            {'type': 'cron', 'spec': '* * * * * 45'},
            {'spec': '23:59:00 @ Sun', 'time': '23:59:00 ', 'by': 'Sun'},
            {'spec': '9:30:10', 'time': '9:30:10'},
            {'spec': '9:30 @ Mon-Fri', 'time': '9:30 ', 'by': 'Mon-Fri'},
            {'spec': '9:30:22 @ Mon,Fri', 'time': '9:30:22 ', 'by': 'Mon,Fri'},
            {'spec': '9:30 , 15:45 @ Mon-Fri', 'time': '9:30 , 15:45 ', 'by': 'Mon-Fri'},
            {'spec': '9:30, 15:45 @ Mon, Fri', 'time': '9:30, 15:45 ', 'by': 'Mon, Fri'},
            {'spec': '-5Sec', 'seconds': '-5'},
            {'spec': '5Min -5Sec', 'minutes': '5', 'seconds': '-5'},
            {'spec': '1w -1h', 'weeks': '1', 'hours': '-1'},
            {'spec': '4month -5hour', 'months': '4', 'hours': '-5'},
            {'spec': '1week -3hours -10Mins', 'weeks': '1', 'hours': '-3', 'minutes': '-10'},
            {'spec': '2h30m -5Sec', 'hours': '2', 'minutes': '30', 'seconds': '-5'},
            {'type': 'calendar', 'spec': '23:00 liquidation', 'time': '23:00 '},
        ]

        for s, r in zip(specs, res):
            assert _parse_schedule_spec(s) == r
            
    def test_recognize_format(self):
        pass
