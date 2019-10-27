import paddle.fluid as fluid
import paddle
import numpy as np
import os


def ensure_list(obj):
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


class Model:
    def __init__(self, inputs, outputs, model_path=None, name=None, **kwargs):
        """
        初始化
        """
        self.name = name or "model"
        self.inputs = ensure_list(inputs)
        self.outputs = ensure_list(outputs)
        self.inputs_name = [tensor.name for tensor in self.inputs]
        self.outputs_name = [tensor.name for tensor in self.outputs]

        self.executor = None
        self.init_program = None
        self.train_program = None
        self.test_program = None

        self.labels = None
        self.cost = None
        self.losses = None
        self.metrics = None
        self.optimizer = None
        self.feeder = None
        self.predict_feeder = None

        self.model_path = model_path
        self.is_compiled = False

    def save_model(self, folder, model_name=None):
        """
        保存模型，用于增量训练或恢复训练
        """
        if model_name is None:
            model_name = self.name
        fluid.io.save_persistables(executor=self.executor,
                                   dirname=folder,
                                   filename=model_name,
                                   main_program=self.train_program)

    def load_model(self, folder, model_name=None):
        """
        加载模型参数，通常用于训练时的初始化
        """
        if model_name is None:
            model_name = self.name
        fluid.io.load_persistables(executor=self.executor,
                                   dirname=folder,
                                   filename=model_name,
                                   main_program=self.train_program)

    def save_inference_model(self, folder, model_name=None):
        """
        保存模型，用于推理
        """
        if model_name is None:
            model_name = self.name
        fluid.io.save_inference_model(executor=self.executor,
                                      dirname=folder,
                                      model_filename=model_name + ".model",
                                      params_filename=model_name + ".params",
                                      feeded_var_names=[t.name for t in self.inputs],
                                      target_vars=self.outputs)

    @staticmethod
    def load_inference_model(folder, model_name=None, executor=None, **kwargs):
        """
        加载模型，用于推理(predict)
        """
        model = Model(None, None)
        model.executor = executor or fluid.Executor(fluid.CUDAPlace(0))

        model_filename = (model_name + ".model") if model_name else None
        params_filename = (model_name + ".params") if model_name else None

        main_program, feed_target, fetch_targets = fluid.io.load_inference_model(executor=model.executor,
                                                                                 dirname=folder,
                                                                                 model_filename=model_filename,
                                                                                 params_filename=params_filename)
        model.test_program = main_program
        model.inputs = [tensor for tensor in main_program.list_vars() if tensor.name in feed_target]
        model.inputs_name = feed_target
        model.outputs = fetch_targets
        model.outputs_name = [tensor.name for tensor in fetch_targets]

        model.predict_feeder = fluid.DataFeeder(model.inputs, model.executor.place)

        return model

    def compile(self, optimizer, loss=None, metrics=None, executor=None):
        """
        编译模型
        """
        if self.is_compiled:
            return

        self.labels = [fluid.layers.data("label_" + tensor.name, tensor.shape) for tensor in self.outputs]

        loss = ensure_list(loss)
        if len(loss) != len(self.outputs):
            raise ValueError("the number of loss should equal to outputs.")

        self.losses = []
        for i in range(len(loss)):
            self.losses.append(fluid.layers.mean(loss[i](self.outputs[i], self.labels[i])))

        self.cost = fluid.layers.sums(self.losses)

        self.optimizer = optimizer
        self.optimizer.minimize(self.cost)

        self.metrics = []
        metrics = ensure_list(metrics)
        for i in range(min(len(metrics), len(self.outputs))):
            self.metrics.append(metrics[i](self.outputs[i], self.labels[i]))

        self.init_program = fluid.default_startup_program()
        self.train_program = fluid.default_main_program()
        self.test_program = self.train_program.clone(for_test=True)

        self.executor = executor or fluid.Executor(fluid.CUDAPlace(0))
        self.feeder = fluid.DataFeeder(self.inputs + self.labels, self.executor.place)

        self.executor.run(self.init_program)
        self.is_compiled = True

    def fit(self):
        """
        训练模型
        """
        pass

    def fit_dataset(self, dataset, epochs, batch_size, echo_freq=1):
        """
        训练模型
        """
        train_reader = paddle.batch(dataset.train_data, batch_size=batch_size)
        test_reader = paddle.batch(dataset.test_data, batch_size=batch_size)

        fetch_list = self.losses + self.metrics

        for epoch in range(epochs):
            for iter, data in enumerate(train_reader()):
                result = self.executor.run(self.train_program, feed=self.feeder.feed(data),
                                           fetch_list=fetch_list)
                if iter % echo_freq == 0:
                    msg = 'Epoch:%d, Iter:%d' % (epoch, iter)
                    for item in result:
                        msg += ", " + str(np.mean(item))
                    print(msg)

            test_result = {i: [] for i in range(len(fetch_list))}
            for iter, data in enumerate(test_reader()):
                result = self.executor.run(self.test_program, feed=self.feeder.feed(data),
                                           fetch_list=fetch_list)
                for i in range(len(result)):
                    test_result[i].append(np.mean(result[i]))

            msg = '[VAL] Epoch:%d' % (epoch,)
            for item in test_result.values():
                msg += ", " + str(np.mean(item))
            print(msg)
            print()

            if self.model_path:
                if os.path.isdir(self.model_path):
                    os.makedirs(self.model_path, exist_ok=True)
                    self.save_model(self.model_path)
                else:
                    folder = os.path.dirname(self.model_path)
                    model_name = os.path.basename(self.model_path)
                    os.makedirs(folder, exist_ok=True)
                    self.save_model(folder, model_name)

    def evaluate(self):
        """
        评估模型
        """
        pass

    def predict(self, *data):
        """
        预测模型
        """
        data = [tuple(data)]

        feed_data = self.predict_feeder.feed(data)
        predict = self.executor.run(self.test_program, feed=feed_data, fetch_list=self.outputs)

        predict = [item[0] for item in predict]
        return predict

    def predict_batch(self, data):
        """
        预测模型
        """
        data = [item if isinstance(item, tuple) else (item,) for item in data]

        feed_data = self.predict_feeder.feed(data)
        predict = self.executor.run(self.test_program, feed=feed_data, fetch_list=self.outputs)

        return predict

# if __name__ == '__main__':
#     import cv2
#     import numpy as np
#     from data import normalize
#
#     model = SRModel.load_inference_model("/home/killf/dlab/FlyAI/SingleImageSuperResolution4Scale/data/output/model",
#                                          "test")
#
#     lr_image_path = "/home/killf/dlab/FlyAI/SingleImageSuperResolution4Scale/data/input/images/1571295036822245.png"
#     lr_img = cv2.imread(lr_image_path)
#     lr_img = normalize(lr_img)
#     lr_img = np.transpose(lr_img, [2, 0, 1])
#     # lr_img = np.expand_dims(lr_img, 0)
#     lr_img = lr_img.astype(np.float32)
#
#     # ret = model.feeder.feed([(lr_img, 123)])
#     # print(ret)
#
#     ret = model.predict_batch([lr_img])
#     print(ret)
#
#     ret = model.predict(lr_img, lr_img)
#     print(ret)
