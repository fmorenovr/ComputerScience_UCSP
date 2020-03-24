#include "pcd2norm.h"

#include <pcl/features/vfh.h>

bool loadCloudPCDNorm (const string &filename, PointCloud<PointNormal> &cloud){
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile<PointNormal> (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", getFieldsList (cloud).c_str ());

  // Check if the dataset has normals
  vector<pcl::PCLPointField> fields;
  if (getFieldIndex (cloud, "normal_x", fields) == -1)
  {
    print_error ("The input dataset does not contain normal information!\n");
    return (false);
  }
  return (true);
}

void saveCloudVFH (const std::string &filename, const PointCloud<VFHSignature308> &output){
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving PCD with VFH"); print_value ("%s ", filename.c_str ());
  
  io::savePCDFileASCII(filename, output);
  
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}

void computeVFH(const PointCloud<PointNormal>::Ptr &cloud, PointCloud<VFHSignature308> &output){
  // Estimate
  TicToc tt;
  tt.tic ();
  
  // Create the VFH estimation class, and pass the input dataset+normals to it
  VFHEstimation<PointNormal, PointNormal, VFHSignature308> vfh;
  vfh.setSearchMethod (search::KdTree<PointNormal>::Ptr (new search::KdTree<PointNormal>));
  vfh.setInputCloud (cloud);
  vfh.setInputNormals (cloud);
  vfh.compute (output);

  print_highlight ("Computed VFH in "); print_value ("%g", tt.toc ()); print_info (" ms for "); print_value ("%d", output.width * output.height); print_info (" points.\n");
}

bool vfhEstimation(const string &pcdFilename, const string &vfhFilename){
  print_info ("Estimate VFH (308) descriptors using pcl::VFHEstimation.\n");

  // Load the first file
  PointCloud<PointNormal>::Ptr cloud (new PointCloud<PointNormal>);
  if (!loadCloudPCDNorm(pcdFilename, *cloud))
    return (false);

  // Perform the feature estimation
  PointCloud<VFHSignature308> output;
  computeVFH(cloud, output);
  // Save into the second file
  saveCloudVFH(vfhFilename, output);
  return true;
}
