import numpy as np

from sklearn.decomposition import PCA


class FeatureConstructor:

    @staticmethod
    def construct_pca_features(pca_config, x_train, x_valid, x_test, model_data_dict):
        if pca_config["construct_all"]:

            model_data_list = []
            for key, value in model_data_dict.items():
                model_data_list.append(value)

            for i in range(len(model_data_list)):
                for j in range(i, len(model_data_list)):
                    pca = PCA(n_components=pca_config["pca_result_num"])

                    # train

                    tmp_train = np.concatenate((model_data_list[i][0], model_data_list[j][0]), axis=1)
                    principal_component_train = pca.fit_transform(tmp_train)
                    x_train = np.concatenate((x_train, principal_component_train), axis=1)

                    # valid

                    tmp_valid = np.concatenate((model_data_list[i][1], model_data_list[j][1]), axis=1)
                    principal_component_valid = pca.fit_transform(tmp_valid)
                    x_valid = np.concatenate((x_valid, principal_component_valid), axis=1)

                    # test

                    tmp_test = np.concatenate((model_data_list[i][2], model_data_list[j][2]), axis=1)
                    principal_component_test = pca.fit_transform(tmp_test)
                    x_test = np.concatenate((x_test, principal_component_test), axis=1)
        else:
            columns_list = pca_config["construction_list"]

            for column_tuple in columns_list:
                c1 = column_tuple[0]
                c2 = column_tuple[1]
                c1_data = model_data_dict[c1]
                c2_data = model_data_dict[c2]

                pca = PCA(n_components=pca_config["pca_result_num"])

                # train

                tmp_train = np.concatenate((c1_data[0], c2_data[0]), axis=1)
                principal_component_train = pca.fit_transform(tmp_train)
                x_train = np.concatenate((x_train, principal_component_train), axis=1)

                # valid

                tmp_valid = np.concatenate((c1_data[1], c2_data[1]), axis=1)
                principal_component_valid = pca.fit_transform(tmp_valid)
                x_valid = np.concatenate((x_valid, principal_component_valid), axis=1)

                # test

                tmp_test = np.concatenate((c1_data[2], c2_data[2]), axis=1)
                principal_component_test = pca.fit_transform(tmp_test)
                x_test = np.concatenate((x_test, principal_component_test), axis=1)
        return x_train, x_valid, x_test

    @staticmethod
    def construct_statistical_features(config, x_train, x_valid, x_test):

        model_num = len(config["first_layer"])
        num_classes = config["num_classes"]

        train_min, train_max, train_mean, train_std = calculate_statistics(x_train, num_classes, model_num)
        valid_min, valid_max, valid_mean, valid_std = calculate_statistics(x_valid, num_classes, model_num)
        test_min, test_max, test_mean, test_std = calculate_statistics(x_test, num_classes, model_num)

        if config["feature_config"]["pca_statistics"]:
            for i in range(len(train_min)):
                pca = PCA(n_components=config["feature_config"]["pca_result_num"])

                # train
                tmp_train = np.concatenate((np.array(train_min[i]).reshape(-1, 1),
                                            np.array(train_max[i]).reshape(-1, 1),
                                            np.array(train_mean[i]).reshape(-1, 1),
                                            np.array(train_std[i]).reshape(-1, 1)),
                                           axis=1)
                principal_component_train = pca.fit_transform(tmp_train)
                x_train = np.concatenate((x_train, principal_component_train), axis=1)

                # valid
                tmp_test = np.concatenate((np.array(valid_min[i]).reshape(-1, 1), np.array(valid_max[i]).reshape(-1, 1),
                                           np.array(valid_mean[i]).reshape(-1, 1),
                                           np.array(valid_std[i]).reshape(-1, 1)),
                                          axis=1)
                principal_component_valid = pca.fit_transform(tmp_test)
                x_valid = np.concatenate((x_valid, principal_component_valid), axis=1)

                # test

                tmp_test = np.concatenate((np.array(test_min[i]).reshape(-1, 1), np.array(test_max[i]).reshape(-1, 1),
                                           np.array(test_mean[i]).reshape(-1, 1), np.array(test_std[i]).reshape(-1, 1)),
                                          axis=1)
                principal_component_test = pca.fit_transform(tmp_test)
                x_test = np.concatenate((x_test, principal_component_test), axis=1)
        else:
            for i in range(len(train_min)):
                # train

                x_train = np.concatenate((x_train, np.array(train_min[i]).reshape(-1, 1),
                                          np.array(train_max[i]).reshape(-1, 1), np.array(train_mean[i]).reshape(-1, 1),
                                          np.array(train_std[i]).reshape(-1, 1)), axis=1)

                # valid
                x_valid = np.concatenate((x_valid, np.array(valid_min[i]).reshape(-1, 1),
                                          np.array(valid_max[i]).reshape(-1, 1),
                                          np.array(valid_mean[i]).reshape(-1, 1),
                                          np.array(valid_std[i]).reshape(-1, 1)), axis=1)

                # test

                x_test = np.concatenate((x_test, np.array(test_min[i]).reshape(-1, 1),
                                         np.array(test_max[i]).reshape(-1, 1), np.array(test_mean[i]).reshape(-1, 1),
                                         np.array(test_std[i]).reshape(-1, 1)), axis=1)

        return x_train, x_valid, x_test


def calculate_statistics(data, num_classes, model_num):
    mins = [[] for _ in range(num_classes)]
    maxs = [[] for _ in range(num_classes)]
    means = [[] for _ in range(num_classes)]
    stds = [[] for _ in range(num_classes)]

    for i in range(len(data)):
        for j in range(9):
            row_class_data = []
            for k in range(model_num):
                row_class_data.append(data[i][j + k * 9])
            mins[j].append(min(row_class_data))
            maxs[j].append(max(row_class_data))
            means[j].append(np.mean(row_class_data))
            stds[j].append(np.std(row_class_data))

    return mins, maxs, means, stds
