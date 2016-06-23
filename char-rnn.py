import random
import numpy as np
import tensorflow as tf

seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)


def one_hot(v):
    return np.eye(vocab_size)[v]

# Data I/O
data = open("data/tinyshakespeare/input.txt", 'r').read()  # Use this source file as input for RNN
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('Data has %d characters, %d unique.' % (data_size, vocab_size))

# create a map from char to index and index to char
char_to_ix =
ix_to_char =

# Hyper-parameters
hidden_size = 100  # hidden layer's size
seq_length = 25  # number of steps to unroll
learning_rate = 1e-1

# initialize the model and params
inputs =
targets =
init_state =

initializer = tf.random_normal_initializer(stddev=0.1)

with tf.variable_scope("RNN") as scope:
    hs_t = init_state
    ys = []
    for t, xs_t in enumerate(tf.split(0, seq_length, inputs)):
        if t > 0:
            scope.reuse_variables()  # Reuse variables
        Wxh =
        Whh =
        Why =
        bh =
        by =

        hs_t =
        ys_t =
        ys.append(ys_t)

# create the backpropagation and error layers
hprev = hs_t
output_softmax = tf.nn.softmax(ys[-1])  # Get softmax for sampling

outputs = tf.concat(0, ys)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))

# Minimizer
minimizer = tf.train.AdamOptimizer()
grads_and_vars = minimizer.compute_gradients(loss)

# Gradient clipping
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad =
    clipped_grads_and_vars.append((clipped_grad, var))

# Gradient updates
updates = minimizer.apply_gradients(clipped_grads_and_vars)

# Session
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Initial values
n, p = 0, 0
hprev_val = np.zeros([1, hidden_size])

# train the model
while True:
    # Initialize
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev_val = np.zeros([1, hidden_size])
        p = 0  # reset

    # Prepare inputs

    input_vals =
    target_vals =

    input_vals = one_hot(input_vals)
    target_vals = one_hot(target_vals)

    hprev_val, loss_val, _ = #sess.run()

    # sampling
    if n % 500 == 0:
        # Progress
        print('iter: %d, p: %d, loss: %f' % (n, p, loss_val))

        # Do sampling
        sample_length = 200
        start_ix = random.randint(0, len(data) - seq_length)
        sample_seq_ix = [char_to_ix[ch] for ch in data[start_ix:start_ix + seq_length]]
        ixes = []
        sample_prev_state_val = np.copy(hprev_val)

        for t in range(sample_length):
            sample_input_vals = one_hot(sample_seq_ix)
            sample_output_softmax_val, sample_prev_state_val = \
                sess.run([output_softmax, hprev],
                         feed_dict={inputs: sample_input_vals, init_state: sample_prev_state_val})

            ix = np.random.choice(range(vocab_size), p=sample_output_softmax_val.ravel())
            ixes.append(ix)
            sample_seq_ix = sample_seq_ix[1:] + [ix]

        txt = ''.join(ix_to_char[ix] for ix in ixes)
        print('----\n %s \n----\n' % (txt,))

    p += seq_length
    n += 1
