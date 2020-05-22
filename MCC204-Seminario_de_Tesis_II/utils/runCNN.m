function output = runCNN(filelist, layer, gap)
  this_folder = pwd;
  
  save([this_folder '/features/tmp_decaf_filelist.mat'], 'filelist', 'gap', '-v7');
  
  if ~ispc
      unix(['python3 utils/runImageNet.py']);
  else
      dos(['C:\cygwin64\bin\bash.exe -c "cd ' [this_folder '/' folder] ';/usr/bin/python runDecaf.py"'])
  end

  output = load([this_folder '/features/tmp_decaf_output.mat']);

end
