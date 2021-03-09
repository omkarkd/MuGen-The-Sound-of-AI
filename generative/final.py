import os
import json
import music21 as m21
import tensorflow.keras as keras
import numpy as np 

model = ' '
MAPPING_PATH = ' '
SEQUENCE_LENGTH = 64
seed_input = ''

def printinfo(Genre,temp):
    global model 
    global MAPPING_PATH
    global seed_input
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if Genre == (g:='austria'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='czech'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='france'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='hungary'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='yugoslavia'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='netherlands'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='poland'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='switzerland'):
        model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        files = os.listdir(dataset_path)
        files = [dataset_path + file for file in files]
        seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_ballad'):
        g = 'germany_ballad'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_boehme'):
        g = 'germany_boehme'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_dva'):
        g = 'germany_dva'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_erk'):
        g = 'germany_erk'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_fink'):
        g = 'germany_fink'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_kinder'):
        g = 'germany_kinder'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='germany_zuccal'):
        g = 'germany_zuccal'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='china_han'):
        g = 'china_han'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='china_natmin'):
        g = 'china_natmin'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='china_shanxi'):
        g = 'china_shanxi'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    elif Genre == (g:='china_xinhua'):
        g = 'china_xinhua'
        # model = os.path.join(BASE_DIR, 'models', f'{g}_model.h5')
        # MAPPING_PATH = os.path.join(BASE_DIR, 'mappers', f'{g}_mapping.json')
        # dataset_path = os.path.join(BASE_DIR, 'datasets', f'{g}', '')
        # files = os.listdir(dataset_path)
        # files = [dataset_path + file for file in files]
        # seed_input  = open(files[np.random.randint(low = 0, high= len(files))]).read()
    return model, temp, MAPPING_PATH, seed_input


class MelodyGenerator:
    '''A class that wraps the LSTM model and offers utilities to generate melodies.'''

    def __init__(self, model_path,mapping_path,sequence_length=64):
        '''Constructor that initialises TensorFlow model'''

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        self.mapping_path = mapping_path
        self.sequence_length = sequence_length

        with open(mapping_path, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ['/'] * sequence_length


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        '''Generates a melody using the DL model and returns a midi file.
        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return melody (list of str): List with symbols representing a melody
        '''

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == '/':
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilites, temperature):
        '''Samples an index from a probability array reapplying softmax using temperature
        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return index (int): Selected output symbol
        '''
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index
    
    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='new.mid'):
        '''Converts a melody into a MIDI file
        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        '''

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != '_' or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign '_'
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

def main(genre, temp, file_name):
    printinfo(Genre = genre, temp = temp)
    mg = MelodyGenerator(model,MAPPING_PATH)
    seed = seed_input
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, temp)
    print('melody generated!!!!!!!!!!!!!!!!!!!!!!!!!')
    mg.save_melody(melody, file_name=file_name)

# if __name__ == '__main__':
#     genre = input('Enter Genre choice:\t')
#     temp = float(input('Enter length(btw 0 - 1):\t'))
#     printinfo(Genre = genre, temp = temp)
#     mg = MelodyGenerator(model,MAPPING_PATH)
#     seed = seed_input
#     melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, temp)
#     print('melody generated!!!!!!!!!!!!!!!!!!!!!!!!!')
#     mg.save_melody(melody)