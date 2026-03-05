import numpy as np
import itertools

# let y^(i)_hat \in [0, 1] be our prediction (a probability)
# that x^{(i)} is a yes-instance of the class.

# For our logistic classifier, y^{(i)}_hat := \sigma(w^T x^{(i)}) 

# The likelihood of our model classifying x^(i) correctly is 
# the Bernoulli(y^{(i)} ; y^(i)_hat)
# = (\hat{y}^{(i)})^{y^{(i)}} * (1 - \hat{y}^{(i)})^{1 - y^{(i)}}

# Thus the negative log likelihood is then 
# - y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})
    
# This is the objective function of our logistic classifier.
def cross_entropy_loss(pred, y):
    return ( -y * np.log(pred) - (1 - y) * np.log(1 - pred) ).sum()


# A numerically stable implementation of the sigmoid function
# using the "logsumexp" trick
def sigmoid(z):
    # Set c to be the elementwise max of 0 and -z
    c = np.maximum(0, -z)
    
    try:
        return np.exp(-c - np.log(np.exp(-c) + np.exp(-z - c)))
    except RuntimeWarning:
            print('Exponential overflow in sigmoid')


class LogisticRegressor:
    def __init__(self, 
                 batch_size=1,
                 learning_rate=0.001,
                 num_epochs=100,
                 momentum_rate=0,
                 regularization_strength=0,
                 seed=None):
        
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.momentum_rate=momentum_rate
        self.regularization_strength = regularization_strength

        self.rng = np.random.default_rng(seed)
    
    # TODO: implement ADAM
    # TODO: Is it possible to get a fully vectorized implementation
    #       that eliminates the need for the outerloop?
    # TODO: clip predictions to prevent log(0), clip gradients for numerical stability
    def fit(self, X, y, learning_curve=False, add_bias=True):
        if add_bias:
            # Concatenate 1's for bias terms
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        
        self.weights = self.rng.random((X.shape[1], 1))
        momentum = np.zeros(self.weights.shape)

        if(learning_curve):
            errors = []

        # weight update step count
        t = 1

        for epoch_count in range(self.num_epochs):
            # Permutation to apply to our rows for stochasticity
            indices = self.rng.permutation(X.shape[0])

            for batch in itertools.batched(indices, self.batch_size):
                # Convert to ndarray to trigger advanced indexing
                ndbatch = np.asarray(batch)

                # Momentum update
                momentum = (
                    self.momentum_rate * momentum + 
                    (1 - self.momentum_rate) * self.loss_gradient(X[ndbatch], y[ndbatch])
                )

                # Because we initialize momentum to 0, our gradient updates are
                # unduly biased towards 0 in the early update steps.
                # We divide by a corrective term to fix this
                unbiased_momentum = momentum / (1 - self.momentum_rate ** t)

                # Weight update
                self.weights -= self.learning_rate * unbiased_momentum
                
                t += 1


            if(learning_curve):
                try:
                    errors.append(self.loss(X, y))
                except RuntimeWarning:
                    print('Division by 0 in log')

        return errors
    
    # Our loss is the L2-regularized cross entropy loss
    def loss(self, X, y):
        ce_loss = cross_entropy_loss(self.predict(X), y)
        # Exclude bias term from regularization penalty
        reg_penalty = (self.regularization_strength / 2) * np.sum(self.weights[1:] ** 2)
        return ce_loss + reg_penalty
    
    # The gradient of our loss function
    def loss_gradient(self, X, y):
        grad = X.T @ (self.predict(X) - y)
        # Add derivative term for regularization
        grad[1:] += self.regularization_strength * self.weights[1:]
        return grad
    
    # prediction of logistic classifer use the sigmoid function
    def predict(self, X):
        # Concatenate 1 for bias terms
        # X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return sigmoid(X @ self.weights)