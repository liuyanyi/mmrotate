# Copyright (c) OpenMMLab. All rights reserved.
class AngelCoder:
    """Angel Coder for Angel Coded Method such as CSL and DCL.

    This coder encodes angel into coded label and decodes
    coded label back to original angel.

    Args:
        type:
        angel_type:
        category_deg:
        window:
        radius:
    """

    def __init__(self,
                 type,
                 angel_type,
                 category_deg=1,
                 window='gaussian',
                 radius=6):
        self.type = type
        self.angel_type = angel_type
        if angel_type == 'oc':
            self.angel_range = 90
            self.angel_offset = 90
        elif angel_type == 'le90':
            self.angel_range = 180
            self.angel_offset = 90
        elif angel_type == 'le135':
            self.angel_range = 180
            self.angel_offset = 45
        self.category_deg = category_deg
        self.window = window
        self.radius = radius

    # @property
    # def coding_length(self):

    def encode(self, angel_targets):
        if self.type == 'csl':
            return self.csl_encode(angel_targets)
        elif self.type == 'dcl':
            pass
        else:
            raise NotImplementedError

    def decode(self, angel_preds):
        if self.type == 'csl':
            return self.csl_decode(angel_preds)
        elif self.type == 'dcl':
            pass
        else:
            raise NotImplementedError

    def csl_encode(self, angel_targets):
        pass

    def csl_decode(self, angel_preds):
        pass
