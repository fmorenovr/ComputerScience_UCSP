function ensuredir(directory)
    if ~exist(directory, 'dir'), mkdir(directory); end
end
