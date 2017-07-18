import tensorflow as tf
from lstm_engine import Lstm_Engine

def main(_):
    print('LSTM Project')
    lstm_engine = Lstm_Engine()
    #lstm_engine.train()
    lstm_engine.run()
    
    
if '__main__' == __name__:
    tf.app.run()