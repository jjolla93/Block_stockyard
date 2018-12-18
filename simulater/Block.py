import datetime

class Block:
    ISROTATED=False

    def __init__(self, width, height, startdate, enddate, name, fixed=-1):
        self._height = height
        self._width = width
        self._startdate = startdate
        self._enddate = enddate
        self.name=name
        self.isin=False
        self.fixed = fixed
        if isinstance(startdate, datetime.date):
            self.term = float((enddate-startdate).days)
        else:
            self.term = enddate - startdate

    def get_dimension(self):
        return self._width, self._height

    def get_schedule(self):
        return self._startdate, self._enddate

    def set_location(self, xlocation, ylocation):
        self._xlocation = xlocation
        self._ylocation = ylocation

    def get_location(self):
        return self._xlocation, self._ylocation