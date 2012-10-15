import pickle

def save_model(filename, model):
    output = open(filename, 'wb')
    pickle.dump(model, output)
    output.close()
    
def load_model(filename):
    pkl_file = open(filename, 'rb')
    res = pickle.load(pkl_file)
    pkl_file.close()
    return res
