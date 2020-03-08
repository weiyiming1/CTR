from model import Model


if __name__ == '__main__':
    model = Model('deepfm')
    train_input, train_label, validate_input, validate_label, test_input, test_label = model.dataset()
    model.train(train_input, train_label, validate_input, validate_label)
    test_label_predict = model.predict(test_input)

