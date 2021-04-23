function [X,grp,coord,labels]= fea2x(fdatabase, idx, dFea)

    total = sum(fdatabase.num_points(idx));
    X = zeros(total, dFea);
    grp = sparse(length(idx), total)';
    labels = zeros(length(idx), 1);
    coord = zeros(total, 2);

    current_start = 0;
    for jj = 1:length(idx)
        %        jj
        current_end = current_start + fdatabase.num_points(idx(jj));
        fpath = fdatabase.path{idx(jj)};
        load(fpath, 'feaSet', 'label');
        X(current_start + 1 : current_end, :) = feaSet.feaArr';
        grp(current_start + 1: current_end, jj) = 1;
        coord(current_start + 1: current_end, 1) = feaSet.x / feaSet.width;
        coord(current_start + 1: current_end, 2) = feaSet.y / feaSet.height;
        labels(jj) = label;
        current_start = current_end;
    end

    grp = grp';
