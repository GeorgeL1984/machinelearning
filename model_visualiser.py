import math
import numpy as np
import matplotlib.pyplot as plt


class ModelVisualiser:
    @staticmethod
    def visualise(model):
        # Plot the samples in a scatter plot and draw the boundary line.

        # Reduce font size for plot title.
        plt.rcParams.update({'font.size': 9})

        # Create figure which will contain 1 or more training/test set pairs of scatter plots.
        fig = plt.figure()

        feature_pairs = []

        # Determine all possible permutations of feature pairs.
        # e.g. for feature_indexes [0, 1, 2, 3] the feature_pairs will be:
        # [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        dim_idx_1_offset = 1
        for dim_idx_0 in range(0, len(model.feature_indexes) - 1):
            for dim_idx_1 in range(dim_idx_1_offset, len(model.feature_indexes)):
                feature_pairs.append([dim_idx_0, dim_idx_1])

            dim_idx_1_offset += 1

        n_plot_pairs_per_figure = 2  # 2 training/test plot pairs in one figure.

        n_plots = np.amin([n_plot_pairs_per_figure * 2, len(feature_pairs) * 2])
        n_plot_cols = 2
        n_plot_rows = math.ceil(n_plots / n_plot_cols)

        plot_idx = 1

        for dim_idx in range(len(feature_pairs)):
            # Create scatter plot for training data.

            feature_pair = feature_pairs[dim_idx]
            plot_pos = [n_plot_rows, n_plot_cols, plot_idx]
            plot_idx += 1

            ModelVisualiser.create_scatter_plot(fig, model.X_train, model.y_train, model.weights, model.bias,
                                                feature_pair, model.feature_names, plot_pos, model.label_values,
                                                model.label_name, 'Training Set', model.train_accuracy)

            # Create scatter plot for test data.

            plot_pos = [n_plot_rows, n_plot_cols, plot_idx]
            plot_idx += 1

            ModelVisualiser.create_scatter_plot(fig, model.X_test, model.y_test, model.weights, model.bias,
                                                feature_pair, model.feature_names, plot_pos, model.label_values,
                                                model.label_name, 'Test Set', model.test_accuracy)

            is_last_feature_pair = (dim_idx + 1) == len(feature_pairs)

            # Show the figure once all the plots for the figure have been created.
            if (dim_idx + 1) % n_plot_pairs_per_figure == 0 or is_last_feature_pair:
                plt.subplots_adjust(wspace=0.4, hspace=0.5)
                plt.show()

                # Create a new figure if all the feature pairs have not been plotted.
                if not is_last_feature_pair:
                    fig = plt.figure()
                    plot_idx = 1

    @staticmethod
    def create_scatter_plot(fig, X, y, weights, bias, dims, feature_names, plot_pos, label_values, label_name, subtitle,
                  accuracy):
        x0_dim, x1_dim = dims
        plot_nrows, plot_ncols, plot_idx = plot_pos

        # Create subblot.
        ax = fig.add_subplot(plot_nrows, plot_ncols, plot_idx)

        # Separate data into two collections, one for each output value (e.g. one collection for Tyrannosaurus and
        # one for Stegossaurus).
        Xy_0 = np.array([np.concatenate((x_row, [y[x_index]])) for x_index, x_row in enumerate(X) if y[x_index] == 0])
        Xy_1 = np.array([np.concatenate((x_row, [y[x_index]])) for x_index, x_row in enumerate(X) if y[x_index] == 1])

        # Plot data for each output value (e.g. blue for Tyrannosaurus and green Stegossaurus).
        plt.scatter(Xy_0[:, x0_dim], Xy_0[:, x1_dim], marker='.', c='blue')
        plt.scatter(Xy_1[:, x0_dim], Xy_1[:, x1_dim], marker='.', c='green')

        # Draw legend.
        plt.legend(label_values)

        # Determine start and end coordinates of boundary line.
        x0_min = np.amin(X[:, x0_dim])
        x0_max = np.amax(X[:, x0_dim])

        line_x_1 = x0_min
        line_x_2 = x0_max

        line_y_1 = (-weights[x0_dim] * x0_min - bias) / weights[x1_dim]
        line_y_2 = (-weights[x0_dim] * x0_max - bias) / weights[x1_dim]

        line_coord_x = [line_x_1, line_x_2]
        line_coord_y = [line_y_1, line_y_2]

        # Draw boundary line.

        ax.plot(line_coord_x, line_coord_y, 'k')

        # Trim vertical white space (i.e not containing scatter plot points).

        # x1_min = np.amin(X[:, x1_dim])
        # x1_max = np.amax(X[:, x1_dim])
        # ax.set_ylim([x1_min, x1_max])

        # Add labels to the axes.

        plt.xlabel(feature_names[x0_dim])
        plt.ylabel(feature_names[x1_dim])

        # Add title.

        plt.title(label_name + ' ' + feature_names[x1_dim] + ' vs ' + feature_names[x0_dim]
                  + '\n' + subtitle + ' [Accuracy: {0:.4f}]'.format(accuracy))
