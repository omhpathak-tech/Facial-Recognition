from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer 

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha 
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha':self.alpha,
        })
        return config
    def triplet_loss(self, inputs):
        a, p, n = inputs 
        p_dist = K.sum(K.square(a-p), axis = -1)
        n_dist = K.sum(K.square(a - n), axis = -1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis = 0)
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss