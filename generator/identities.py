"""
Class representing user and generator identities
for use by speaker tracking.
"""

class Identities(object):
    def __init__(self):
        self.user = "User"
        self.generator = "Generator"
        self.is_swapped = False

    def swap(self):
        tmp = self.user
        self.user = self.generator
        self.generator = tmp
        self.is_swapped = not self.is_swapped
        
    def reset(self):
        if self.is_swapped:
            self.swap()