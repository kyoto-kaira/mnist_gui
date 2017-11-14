class OneLineInfo:
    def __init__(self):
        self.Destination = None

    def set_destination(self, func):
        self.Destination = func

    def send(self, text):
        if self.Destination is None:
            print(text)
        else:
            self.Destination(text)


global_one_line_info = OneLineInfo()
