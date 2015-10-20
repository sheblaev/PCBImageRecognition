from pyfann import libfann

num_input = 432
num_output = 10
num_layers = 3
num_neurons_hidden = 144
desired_error = 0.00006
max_epochs = 50000
epochs_between_reports = 1000

ann = libfann.neural_net()

ann.create_standard(num_layers, num_input, num_neurons_hidden, num_output)
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file('samples.txt', max_epochs, epochs_between_reports, desired_error)

ann.save('fann.data')
ann.destroy()

        """
        def MagicRegognition(img, ann):
                ann = libfann.neural_net()
                ann.create_from_file('fann.data')

                sample = []
                for i in img.size[1]:
                        for j in img.size[0]:
                                if colordist(img.getpixel((j, i)), bgcolor) < 10:
                                        sample[j + i * img.size[0]] = 0
                                else:
                                        sample[j + i * img.size[0]] = 1

                res = ann.run(sample)

                return res.index(max(res))
        
        """

