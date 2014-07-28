import sys, math, collections, subprocess, string

#Reads model into a map from feature strings to weights
def map(model):
    f = open(model, 'r')
    #Initializes v as a dictionary
    v = collections.defaultdict(int)
    #For each feature in the model, sets feature as the key and weight as the value in v dictionary
    for line in f:
        l = line.split()
        v[l[0]] = float(l[1])
    f.close()
    #Returns dictionary of feature:weight values
    return v

#Helper function to call provided scripts as subprocesses
def process(args):
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

#Helper function to generate output from subprocesses
def call(process, args):
    process.stdin.write(args + '\n')
    line = ''
    while True:
        l = process.stdout.readline()
        if not l.strip(): 
            break
        line += l
    return line

#Calculates features from histories and sentence, calculates feature weights given v from trained model
def calc_features(histories, sentence, v):
    #Initializes scores as an empty string
    scores = ''
    for history in histories:
        h = history.split()
        #If history is non-empty and is not the end of the sentence
        if len(h) > 0 and h[2] != 'STOP':
            #Sets first position as index 0 to correspond with sentence indices
            pos = int(h[0]) - 1
            word = sentence[pos]
            tag = h[2]
            #Sets initial weight to 0
            weight = 0
            #Creates set of possible features from current history
            features = ['BIGRAM:' + h[1] + ':' + tag, 'TAG:' + word + ':' + tag]
            #For each feature in the set, calculates the weight using v
            for feature in features:
                #Checks if feature is in v; if so, sets weight using feature weight from v
                if feature in v:
                    weight += v[feature]
            #Appends history and weight to scores
            scores += (history + ' ' + str(weight) + '\n')
    return scores

#Takes a v vector, a development file, and the name of the designated output file, computes highest scoring tags and writes to output file 
def tag(v, input_file, output_file):
    enumerate_histories = process(['python', 'tagger_history_generator.py', 'ENUM'])
    score_histories = process(['python', 'tagger_decoder.py', 'HISTORY'])
    ifile = open(input_file, 'r')
    ofile = open(output_file, 'w')
    #Initializes empty sentence
    sentence = ''
    #For each line in the input file
    for line in ifile:
        l = line.strip()
        #If line is not empty
        if l:
            #Add line to sentence 
            sentence += line
        #If line is empty, it means we are at the end of the sentence
        else: 
            #Enumerates possible histories for the sentence
            histories = call(enumerate_histories, sentence).split('\n')
            sentence = sentence.split()
            #Calculates weights for all histories
            scores = calc_features(histories, sentence, v)
            #Finds highest scoring sequence of histories
            tags = call(score_histories, scores).split('\n')
            #Writes (word, tag) pairs to output file
            for i in range(len(sentence)): 
                tag = tags[i].split()[2]
                word = sentence[i]
                ofile.write(word + ' ' + tag + '\n')
            #Inserts empty line between sentences
            ofile.write('\n')
            #Resets empty sentence
            sentence = ''
    ofile.close()
    ifile.close()

def train_model(training_data):
    gold_tagger = process(['python', 'tagger_history_generator.py', 'GOLD'])
    enumerate_histories = process(['python', 'tagger_history_generator.py', 'ENUM'])
    score_histories = process(['python', 'tagger_decoder.py', 'HISTORY'])
    v = collections.defaultdict(int)
    g = {}
    for iteration in range(5):
        f = open(training_data, 'r')
        sentence = ''
        for line in f:
            l = line.strip()
            #If line is not empty
            if l:
                #Add line to sentence 
                sentence += line
            else:
                histories = call(enumerate_histories, sentence).split('\n')
                sentence = sentence.split()
                scores = calc_features(histories, sentence, v)
                tags = call(score_histories, scores).split('\n')

def suffix(word, j):