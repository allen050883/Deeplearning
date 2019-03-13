import tensorflow as tf

def model():
  opt = tf.train.AdamOptimizer()
  tvs = tf.trainable_variables()
  accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
  zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
  gvs = opt.compute_gradients(rmse, tvs)
  accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
  train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
pass

def training():
  sess.run(zero_ops)
  for i in xrange(n_minibatches):
      sess.run(accum_ops, feed_dict=dict(X: Xs[i], y: ys[i]))
  sess.run(train_step)
