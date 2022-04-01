import mitdeeplearning as mdl

class Songs:
    def __init__(self):
        self.songs = mdl.lab1.load_training_data()
    def load_custom(self, songs):
        self.songs = songs
    def get_songs(self):
        print(self.songs)
    def get_song(self, idx):
        print(self.songs[idx])