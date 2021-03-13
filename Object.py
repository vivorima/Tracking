class Object:

    def __init__(self, object_id, color):
        self.object_id = object_id
        self.color = color
        self.frames = {}

    def addColor(self, color):
        self.color = color

    def set_id(self, id):
        self.object_id = id

    def get_color(self):
        return self.color

    def get_id(self):
        return self.object_id

    def appears_frame(self, frame, position):
        self.frames[frame] = position

    def getFrame(self, frame):
        return self.frames[frame]

    def __str__(self):
        return "Object " + str(self.object_id) + ", appears in frames: " + str(self.frames)

    def __repr__(self):
        return "Object " + str(self.object_id) + ", appears in frames: " + str(self.frames)

    def __eq__(self, other):
        if isinstance(other, Object):
            # We can only compare if `other` is a Shape as well
            return self.object_id == other.object_id
        return NotImplemented

    def get_last_frame(self):
        """gettings coords of the objects last frame"""
        return self.frames[list(self.frames.keys())[-1]]