import time


def load_train(model, x_train, y_train, model_name, model_folder, model_prefix, load_saved):
    model_file = model_folder + model_prefix + model_name

    if load_saved:
        model.initialize()
        model.load_params(model_file)

        return 0
    else:
        start = time.time_ns()
        model.fit(x_train, y_train)
        end = time.time_ns()
        print('fit time ' + model_name + ' %d ns' % (end - start))
        model.save_params(model_file)

        return end - start