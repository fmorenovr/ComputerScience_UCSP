function plot_predictions_on_new_data(config, output_path, title_str, data, test)
    ensuredir(output_path);
    num_pages = floor(length(data.test_images) / 400);
    nav_str = title_str; 
    for pp = 1 : num_pages
        nav_str = sprintf('%s<a href="predictions_%d.html">%d</a> | ', nav_str, pp, pp);
    end
    per_page_count = round(length(data.test_images) / num_pages);
    for pp = 1 : num_pages
        f = fopen([output_path '/predictions_' num2str(pp) '.html'], 'w');   
        fprintf(f, '%s<table><tr>', nav_str); count = 0;
        [s_scores, s_inds] = sort(test.predicted_labels, 'descend');
        for ii = (1 + ((pp - 1) * per_page_count)) : min(pp * per_page_count, length(data.test_images))
            [path, image_name, ext] = fileparts(data.test_images{s_inds(ii)});
            fprintf(f, '<td><table><tr>');
            fprintf(f, '<td><img src="%s/%s%s" height="180"/></td></tr>', ...
                       config.image_url, image_name, ext);
            fprintf(f, '<tr><td>%d. SVR: %2.4f, GT: %2.4f</td>', ii, ...
                       s_scores(ii), test.test_labels(s_inds(ii)));
            fprintf(f, '</tr></table></td>'); count = count + 1;
            if mod(count, 6) == 0, fprintf(f, '</tr><tr>'); end
        end
        fprintf(f, '</tr></table>%s', nav_str);     
        fclose(f);
    end

end
