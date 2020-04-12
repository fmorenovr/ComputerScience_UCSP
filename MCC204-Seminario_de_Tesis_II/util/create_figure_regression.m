function create_figure_regression(labels, predictions, title_str, path)
    train_labels = labels;
    predicted_labels_train = predictions;
    figure;
    [sorted_labels, inds] = sort(train_labels);
    hold on;
    plot(sorted_labels, '.', 'Color', 'blue', 'LineWidth', 2);
    plot(predicted_labels_train(inds), 'x', 'Color', 'red');
    pbaspect([1 1 1]);
    ylim([0 12]);
    title(title_str);
    hold off;
    [fpath, fname, fext] = fileparts(path);
    if ~exist(fpath, 'dir'), mkdir(fpath); end
    print('-dpng', '-r60', path);
    close;
end
