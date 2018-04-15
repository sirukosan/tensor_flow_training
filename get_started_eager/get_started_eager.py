from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    _features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Last field is the label
    _label = tf.reshape(parsed_line[-1], shape=())
    return _features, _label


def loss(_model, _x, _y):
    y_ = _model(_x)
    return tf.losses.sparse_softmax_cross_entropy(labels=_y, logits=y_)


def grad(_model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(_model, inputs, targets)
    return tape.gradient(loss_value, _model.variables)


def get_train_dataset(train_dataset_url):
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url)
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    rv = tf.data.TextLineDataset(train_dataset_fp)
    rv = rv.skip(1)  # skip the first header row
    rv = rv.map(parse_csv)  # parse each row
    rv = rv.shuffle(buffer_size=1000)  # randomize
    rv = rv.batch(32)
    return rv


def get_test_dataset(test_dataset_url):
    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url),
                                      origin=test_dataset_url)

    rv = tf.data.TextLineDataset(test_fp)
    rv = rv.skip(1)  # skip header row
    rv = rv.map(parse_csv)  # parse each row with the funcition created earlier
    rv = rv.shuffle(1000)  # randomize
    rv = rv.batch(32)  # use the same batch size as the training set
    return rv


def plot_training(_train_loss_results, _train_accuracy_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(_train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(_train_accuracy_results)

    plt.show()


tf.enable_eager_execution()

train_dataset = get_train_dataset("http://download.tensorflow.org/data/iris_training.csv")
features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

plot_training(train_loss_results, train_accuracy_results)

test_dataset = get_test_dataset("http://download.tensorflow.org/data/iris_test.csv")

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))
