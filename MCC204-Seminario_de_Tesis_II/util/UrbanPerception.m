% Vicente Ordonez @ UNC Chapel Hill 2013
% UrbanPerception deals with MIT's Place Pulse data.
% addpath(genpath('../lib/util'));
classdef UrbanPerception
    properties
        data = [];
    end
    methods
        function obj = UrbanPerception(data_file, data_format, year)
            obj.data = UrbanPerception.LoadData(data_file, data_format, year);
        end

        function labeled_data = getUnbalancedBinaryData(obj, city_str, metric_str, delta, mode)
            data = obj.getData(city_str);
            if strcmp(mode, 'top')
                [sorted_scores, sorted_inds] = sort(data.(['qs_' metric_str]), 'descend');
            else
                [sorted_scores, sorted_inds] = sort(data.(['qs_' metric_str]), 'ascend');
            end
            cut_point = floor(length(data.ids) * delta);            

            pos_images = data.image_names(sorted_inds(1:cut_point));  % top scored.
            neg_images = data.image_names(sorted_inds(cut_point + 1 : end)); % all others.
            
            labeled_data.images = [pos_images, neg_images]';
            labeled_data.labels = [ones(1, length(pos_images)), -ones(1, length(neg_images))]';
        end

        function labeled_data = getBinaryData(obj, city_str, metric_str, delta)
            data = obj.getData(city_str);
            [sorted_scores, sorted_inds] = sort(data.(['qs_' metric_str]), 'descend');
            cut_point = floor(length(data.ids) * delta);            

            pos_images = data.image_names(sorted_inds(1:cut_point));
            neg_images = data.image_names(sorted_inds(end - cut_point + mod(end, 3): end));
            
            labeled_data.images = [pos_images, neg_images]';
            labeled_data.labels = [ones(1, length(pos_images)), -ones(1, length(neg_images))]';
        end

        function labeled_data = getLabeledData(obj, city_str, metric_str, scale_type)
            data = obj.getData(city_str);
            labeled_data.images = data.image_names;
            labeled_data.latitudes = str2double(data.latitudes);
            labeled_data.longitudes = str2double(data.longitudes);
            labeled_data.ids = data.ids; 
            if nargin > 3 && strcmp(scale_type, 'normalized')
                labels_ = data.(['qs_' metric_str]);
                labeled_data.labels = 10 ./ (1 + exp(-(labels_ - mean(labels_))));
            else
                labeled_data.labels = data.(['qs_' metric_str]);
            end
        end

        function data = getData(obj, city_str)
            inds = strcmp(obj.data.cities, city_str);
            data.qs_safer = obj.data.qs_safer(inds);
            data.qs_safer_error = obj.data.qs_safer_error(inds);
            data.qs_unique = obj.data.qs_unique(inds);
            data.qs_unique_error = obj.data.qs_unique_error(inds);
            data.qs_upperclass = obj.data.qs_upperclass(inds);
            data.qs_upperclass_error = obj.data.qs_upperclass_error(inds);
            data.ids = obj.data.ids(inds);
            data.headings = obj.data.headings(inds);
            data.pitches = obj.data.pitches(inds);
            data.longitudes = obj.data.longitudes(inds);
            data.latitudes = obj.data.latitudes(inds);
            data.image_names = obj.data.image_names(inds);
        end
        
        function plotData(obj, city_str, metric_str, output_str, image_url)
            city_id = regexprep(lower(city_str), ' ', '_');
            output_str = [output_str '/' metric_str];
            if ~exist(output_str, 'dir'), mkdir(output_str); end
            data = obj.getData(city_str);
            scores = data.(['qs_' metric_str]);
            scores_errors = data.(['qs_' metric_str '_error']);
            
            [sorted_scores, sorted_inds] = sort(scores);
            sorted_scores_errors = scores_errors(sorted_inds);
            figure;
            plot(sorted_scores, 'Color', 'blue');hold;
            plot(sorted_scores_errors, 'Color', 'red');
            legend([metric_str ' scores'], [metric_str ' scores errors'], 'Location', 'NorthWest');
            print('-dpng', '-r60', sprintf('%s/scores_fig_%s.png', output_str, city_id));
            close;

            [sorted_scores, sorted_inds] = sort(scores, 'descend');
            sorted_scores_errors = scores_errors(sorted_inds);
            results_per_page = 50;
            n_pages = ceil(length(scores) / results_per_page);
            nav_str = '';
            for ii = 1 : n_pages
                nav_str = sprintf('%s | <a href="p_%s_%d">%02d</a>', nav_str, city_id, ii, ii);
            end
            for ii = 1 : n_pages
                f = fopen(sprintf('%s/p_%s_%d.html', output_str, city_id, ii), 'w');
                fprintf(f, '<html><body><h2>%s (%s scores)</h2>', city_str, metric_str);
                fprintf(f, '<img src="scores_fig_%s.png"/></br>', city_id);
                fprintf(f, '\n%s\n<table border>', nav_str);
                for jj = 1 + (ii - 1) * results_per_page : min(length(scores), ii * results_per_page)
                    fprintf(f, '<tr><td>%d</td>', jj);
                    fprintf(f, '<td><img src="%s/%s" height="180"/></td>', image_url, data.image_names{sorted_inds(jj)});
                    fprintf(f, '<td>%.2f (error = %.2f)</td></tr>', ...
                            sorted_scores(jj), sorted_scores_errors(jj)); 
                end
                fprintf(f, '</table>%s</body></html>', nav_str);
                fclose(f);
            end
        end
    end
   
    methods(Static)
        function data = ListCities()
            data = {'New York City', 'Boston'};
            % data = {'New York City', 'Boston', 'Linz', 'Salzburg'};
        end
        function data = ListCitiesZoomLevels()
            data = [11, 12];
        end
        function data = LoadData(data_file, filetype, year)
            if strcmp(filetype, 'json')
                % {"QS Upperclass": "4.31", "Error in QS Upperclass": "0.47", ...
                %  "Lon": "14.309", "File_Location": "/images/id_1867_400_300.jpg", ...
                %  "Pitch": "NULL", "Error in QS Unique": "0.23", "Heading": "NULL", ...
                %  "City": "Linz", "QS Unique": "3.39", "QS Safer": "4.31", ...
                %  "Lat": "48.271", "ID": "1867", "Error in QS Safer": "0.53"}

                f = fopen(data_file);
                str = fscanf(f, '%c', inf);

                urban.qs_safer = regexp(str, '"QS Safer": "[^"]*"', 'match');
                urban.qs_safer = cellfun(@(x){x(length('"QS Safer": "') + 1: end - 1)}, urban.qs_safer);
                urban.qs_safer_error = regexp(str, '"Error in QS Safer": "[^"]*"', 'match');
                urban.qs_safer_error = cellfun(@(x){x(length('"Error in QS Safer": "') + 1: end - 1)}, urban.qs_safer_error);
                urban.qs_safer = str2double(urban.qs_safer);
                urban.qs_safer_error = str2double(urban.qs_safer_error);

                urban.qs_unique = regexp(str, '"QS Unique": "[^"]*"', 'match');
                urban.qs_unique = cellfun(@(x){x(length('"QS Unique": "') + 1: end - 1)}, urban.qs_unique);
                urban.qs_unique_error = regexp(str, '"Error in QS Unique": "[^"]*"', 'match');
                urban.qs_unique_error = cellfun(@(x){x(length('"Error in QS Unique": "') + 1: end - 1)}, ...
                                                urban.qs_unique_error);
                urban.qs_unique = str2double(urban.qs_unique);
                urban.qs_unique_error = str2double(urban.qs_unique_error);

                urban.qs_upperclass = regexp(str, '"QS Upperclass": "[^"]*"', 'match');
                urban.qs_upperclass = cellfun(@(x){x(length('"QS Upperclass": "') + 1: end - 1)}, urban.qs_upperclass);
                urban.qs_upperclass_error = regexp(str, '"Error in QS Upperclass": "[^"]*"', 'match');
                urban.qs_upperclass_error = cellfun(@(x){x(length('"Error in QS Upperclass": "') + 1: end - 1)}, ...
                                                 urban.qs_upperclass_error);
                urban.qs_upperclass = str2double(urban.qs_upperclass);
                urban.qs_upperclass_error = str2double(urban.qs_upperclass_error);

                urban.latitudes = regexp(str, '"Lat": "[\-0-9\.]+"', 'match');
                urban.latitudes = cellfun(@(x){x(length('"Lat": "') + 1: end - 1)}, urban.latitudes);

                urban.longitudes = regexp(str, '"Lon": "[\-0-9\.]+"', 'match');
                urban.longitudes = cellfun(@(x){x(length('"Lon": "') + 1: end - 1)}, urban.longitudes);

                urban.pitches = regexp(str, '"Pitch": "[^"]*"', 'match');
                urban.pitches = cellfun(@(x){x(length('"Pitch": "') + 1: end - 1)}, urban.pitches);

                urban.headings = regexp(str, '"Heading": "[^"]*"', 'match');
                urban.headings = cellfun(@(x){x(length('"Heading": "') + 1: end - 1)}, urban.headings);

                urban.cities = regexp(str, '"City": "[^"]*"', 'match');
                urban.cities = cellfun(@(x){x(length('"City": "') + 1: end - 1)}, urban.cities);

                urban.ids = regexp(str, '"ID": "[^"]*"', 'match');
                urban.ids = cellfun(@(x){x(length('"ID": "') + 1: end - 1)}, urban.ids);
                
                urban.image_names = cellfun(@(x){sprintf('%s.jpg', x)}, urban.ids);
            else
                % This is a csv file.
                f = fopen(data_file);
                data = textscan(f, '%s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter', ',', 'Headerlines', 1);
                fclose(f);
                urban.qs_safer = str2double(data{2})';
                urban.qs_safer_error = str2double(data{3})';
                urban.qs_unique = str2double(data{4})';
                urban.qs_unique_error = str2double(data{5})';
                urban.qs_upperclass = str2double(data{6})';
                urban.qs_upperclass_error = str2double(data{7})';
                urban.latitudes = data{8}';
                urban.longitudes = data{9}';
                urban.cities = data{10}';
                urban.headings = data{11}';
                urban.pitches = data{12}';
                urban.ids = data{1}';
                urban.image_names = cellfun(@(x){x(9:end)}, data{13}');
            end
            % Exclude cities not returned in the ListCities static method.
            data = urban;
            cities = UrbanPerception.ListCities();
            good_inds = zeros(1, length(urban.cities));
            for t = 1 : length(cities)
                good_inds = good_inds + strcmp(urban.cities, cities{t});
            end
            bad_inds = ~(good_inds > 0);

            invalids = isnan(data.qs_safer) | isnan(data.qs_unique) | isnan(data.qs_upperclass) | bad_inds;
            data.qs_safer = data.qs_safer(~invalids);
            data.qs_safer_error = data.qs_safer_error(~invalids);
            data.qs_unique = data.qs_unique(~invalids);
            data.qs_unique_error = data.qs_unique_error(~invalids);
            data.qs_upperclass = data.qs_upperclass(~invalids);
            data.qs_upperclass_error = data.qs_upperclass_error(~invalids);
            data.latitudes = data.latitudes(~invalids);
            data.longitudes = data.longitudes(~invalids);
            data.pitches = data.pitches(~invalids);
            data.headings = data.headings(~invalids);
            data.cities = data.cities(~invalids);
            data.ids = data.ids(~invalids);
            data.image_names = data.image_names(~invalids);
        end
    end
end

