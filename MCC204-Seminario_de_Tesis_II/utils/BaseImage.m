% Vicente Ordonez @ UNC Chapel Hill.
% BaseImage: Encapsulates helper functions for generic RGB image data.
classdef BaseImage < handle
    properties
        original_width = 0;
        original_height = 0;
        image = [];  % original image in double precision.
        width = 0;
        height = 0;
        spixels = [];  % super pixel index image.
    end
    
    methods
        % Read image and convert to double image.
        function inst = BaseImage(image_fname)
			if ischar(image_fname)
            	inst.image = im2double(imread(image_fname));
			else
				inst.image = im2double(image_fname);
			end
            inst.original_width = size(inst.image, 2);
            inst.original_height = size(inst.image, 1);
            inst.width = inst.original_width;
            inst.height = inst.original_height;
            if size(inst.image, 3) == 1  % Represent grayscale as RGB.
                inst.image = repmat(inst.image, [1 1 3]);
            end
        end
        % Resize image by using a factor.
        function resize(obj, factor, varargin)
            if nargin == 2
                image_height = round(factor * obj.original_height);
                image_width = round(factor * obj.original_width);
            else
                image_height = factor;
                image_width = varargin{1};
            end
            obj.image = imresize(obj.image, [image_height image_width]);
            obj.image(obj.image > 1) = 1;
            obj.image(obj.image < 0) = 0;
            obj.height = image_height;
            obj.width = image_width;
        end
        % Resize image by specifying a max dimension size.
        function resize_max(obj, max_dimension)
            max_dimsize = max(obj.original_height, obj.original_width);
            factor = max_dimension / max_dimsize;
            obj.resize(factor);
        end
        % Resize image by specifying a min dimension size.
        function resize_min(obj, min_dimension)
            min_dimsize = min(obj.original_height, obj.original_width);
            factor = min_dimension / min_dimsize;
            obj.resize(factor);
        end
        % Mask the image with a white background.
        function masked = maskit(obj, mask)
            mask = imresize(im2double(mask), [size(obj.image, 1) size(obj.image, 2)]);
            masked = im2double(obj.image) .* repmat(mask, [1 1 3]);
            masked = masked + (1 - repmat(mask, [1 1 3]));
        end
        % Overlay heatmap
        function heatmap_overlay = heatmap_overlay(obj, mask)
            cmap = jet(255); cmap(1:25, 3) = 0; % Threshold at some reasonble point for visualization.
            rgb_mask = ind2rgb(gray2ind(mask, 255), cmap);
            heatmap_overlay = 0.6 * rgb_mask + 0.8 * obj.image;
        end

		function crop_img = crop(obj, box)
			crop_img = obj.image(box(1):box(3), box(2):box(4), :);
		end

		% overlay rectangles on top of the image.
		function image_out = show_boxes(obj, boxes)
			im = obj.image;
			for i = 1 : size(boxes, 1)
				box = boxes(i, :);
				im(box(1):box(3), box(2), 1) = 1;
				im(box(1):box(3), box(2), 2) = 0;
				im(box(1):box(3), box(2), 3) = 0;
				
				im(box(1):box(3), box(4), 1) = 1;
				im(box(1):box(3), box(4), 2) = 0;
				im(box(1):box(3), box(4), 3) = 0;
				
				im(box(1), box(2):box(4), 1) = 1;
				im(box(1), box(2):box(4), 2) = 0;
				im(box(1), box(2):box(4), 3) = 0;
				
				im(box(3), box(2):box(4), 1) = 1;
				im(box(3), box(2):box(4), 2) = 0;
				im(box(3), box(2):box(4), 3) = 0;
			end
			image_out = im;
		end

        % Show image.
        function show(obj)
            imshow(obj.image);
        end
        % Save image.
        function save(obj, image_fname)
            BaseImage.imwrite(obj.image, image_fname);
        end
    end
    
    methods(Static)
        % Save image and make sure directory exists or is created.
        function imwrite(image, image_fname)
            [directory, ~, ~] = fileparts(image_fname);
            if ~exist(directory, 'dir')
                mkdir(directory);
            end
            imwrite(image, image_fname);
        end
        % Utility function for rendering html with clipped image sizes 
        % sprintf('<img src="http://foo.org/im.jpg" %s>', ...
        %         BaseImage.html_dim(bimage, 200, 200));
        function str = html_dim(obj, max_height, max_width)
            if obj.width < obj.height
                str = sprintf('height=%d', max_height);
            else
                str = sprintf('width=%d', max_width);
            end
        end
    end
end
