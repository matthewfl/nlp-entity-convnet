import theano.tensor as T
import lasagne
import theano


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=100,
                          learning_rate=.01, momentum=.9):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    output = lasagne.layers.get_output(output_layer, X_batch)
    loss_train = lasagne.objectives.categorical_crossentropy(output, y_batch)
    loss_train = loss_train.mean()

    output_test = lasagne.layers.get_output(output_layer, X_batch,
                                            deterministic=True)
    loss_eval = lasagne.objectives.categorical_crossentropy(output_test,
                                                            y_batch)
    loss_eval = loss_eval.mean()

    pred = T.argmax(output_test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    # iter_valid = theano.function(
    #     [batch_index], [loss_eval, accuracy],
    #     givens={
    #         X_batch: dataset['X_valid'][batch_slice],
    #         y_batch: dataset['y_valid'][batch_slice],
    #     },
    # )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        # valid=iter_valid,
        test=iter_test,
    )
