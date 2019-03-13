import tensorflow as tf

def model():
    ## Optimizer definition - nothing different from any classical example
    opt = tf.train.AdamOptimizer()

    ## Retrieve all trainable variables you defined in your graph
    tvs = tf.trainable_variables()
    ## Creation of a list of variables with the same shape as the trainable ones
    # initialized with 0s
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
    gvs = opt.compute_gradients(rmse, tvs)

    ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    ## Define the training step (part with variable value update)
    train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
pass


def training():
      # Run the zero_ops to initialize it
      sess.run(zero_ops)
      # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
      for i in xrange(n_minibatches):
          sess.run(accum_ops, feed_dict=dict(X: Xs[i], y: ys[i]))
      # Run the train_step ops to update the weights based on your accumulated gradients
      sess.run(train_step)
pass
