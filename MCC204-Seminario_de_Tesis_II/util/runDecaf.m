function output = runDecaf(filelist, layer)
  this_folder = pwd;
  
  save([this_folder '/features/tmp_decaf_filelist.mat'], 'filelist', '-v7');
  
  if ~ispc
      unix(['python3 util/runImageNet.py']);
  else
      dos(['C:\cygwin64\bin\bash.exe -c "cd ' [this_folder '/' folder] ';/usr/bin/python runDecaf.py"'])
  end

  output = load([this_folder '/features/tmp_decaf_output.mat']);

end
