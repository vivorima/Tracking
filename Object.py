from decimal import Decimal


class Object:

    colors = [(43, 153, 254), (0, 103, 255), (121, 200, 0), (113, 67, 115), (220, 182, 50), (204, 37, 49),
              (49, 37, 204), (29, 112, 68), (0, 103, 255)]

    def __init__(self, object_id, bounding):

        # randomize color of all the objects boxes

        self.object_id = object_id
        self.color = Object.colors[object_id-1]
        self.x, self.y, self.w, self.h = bounding

        self.centroid = [self.x + (self.w / 2),
                         self.y + (self.h / 2)]

        self.ratio = float(int(100 * (self.w / self.h)))/100

    def set_id(self, id):

        self.object_id = id
        self.color = Object.colors[id-1]

    def get_centroid(self):
        return self.centroid

    def get_color(self):
        return self.color

    def get_id(self):
        return self.object_id

    def get_x(self):
        return self.x

    def get_ratio(self):
        return self.ratio

    def get_y(self):
        return self.y

    def get_w(self):
        return self.w

    def get_h(self):
        return self.h

    def __str__(self):
        return "Object " + str(self.object_id) + "x = " + str(self.x) + "y = " + str(self.y) + "w = " + str(self.w) + "h = " + str(self.h) + " ratio " + str(self.ratio)

    def __repr__(self):
        return "Object " + str(self.object_id) + "x = " + str(self.x) + "y = " + str(self.y) + "w = " + str(self.w) + "h = " + str(self.h) + " ratio " + str(self.ratio)
