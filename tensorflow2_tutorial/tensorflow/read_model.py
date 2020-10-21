# -------------------------
# -----  Toy Context  -----
# -------------------------
import tensorflow as tf


class Net(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10.0)[:, None]
    labels = inputs * 5.0 + tf.range(5.0)[None, :]
    return (
        tf.data.Dataset.from_tensor_slices(dict(x=inputs, y=labels)).repeat().batch(2)
    )


def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example["x"])
        loss = tf.reduce_mean(tf.abs(output - example["y"]))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

# ---------------------
# -----  Restore  -----
# ---------------------

# In another script, re-initialize objects
opt = tf.keras.optimizers.Adam(0.1)
net = Net()
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator
)
manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)

# Re-use the manager code above ^

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

for _ in range(50):
    example = next(iterator)
    # Continue training or evaluate etc.
