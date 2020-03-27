#include "pcd2visualize.h"

bool visualizePCDs = false;

struct PlySaver{
  PlySaver(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud, bool binary, bool use_camera, K2G & k2g): cloud_(cloud), binary_(binary), use_camera_(use_camera), k2g_(k2g){}

  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud_;
  bool binary_;
  bool use_camera_;
  K2G & k2g_;
};

void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void * data){
  string pressed = event.getKeySym();
  string now;
  PlySaver * s = (PlySaver*)data;
  if(event.keyDown ()){
    if(pressed == "s"){
      chrono::high_resolution_clock::time_point p = chrono::high_resolution_clock::now();
      now = to_string((long)chrono::duration_cast<chrono::milliseconds>(p.time_since_epoch()).count());
      
      // Generate point clouds in ply file
      saveCloudPLY(pathPCL+"pc_"+now+".ply", *(s->cloud_), s->binary_, s->use_camera_);
      
      // convert ply to pcd
      if(PLY2PCD(pathPCL+"pc_"+now+".ply", pathPCL+"pcd_"+now+".pcd", false)){
        // visualize PCD
        visualizePCD(pathPCL+"pcd_"+now+".pcd", "Point Cloud Data Viewer at "+now);
        // normal estimation
        if(normalEstimation(pathPCL+"pcd_"+now +".pcd", pathPCL+"pcd_"+now+"_norm.pcd", 0, 0.03, false)){
          // visualize PCD Norm
          visualizePCD(pathPCL+"pcd_"+now+"_norm.pcd", "Point Cloud Data Normalize Viewer at "+now);
          // vfh Estimation
          if(vfhEstimation(pathPCL+"pcd_"+now+"_norm.pcd", pathPCL+"pcd_"+now+"_vfh.pcd")){
            // visualize VFH
            visualizePCD(pathPCL+"pcd_"+now+"_vfh.pcd", "Point Cloud Data VFH Viewer at "+now);
          }
        }
      }
    }
    if(pressed == "x") {
        s->k2g_.storeParameters();
        cout << "stored calibration parameters" << endl;
    }
    if(pressed == "m"){
      s->k2g_.mirror();
    }
#ifdef WITH_SERIALIZATION
    if(pressed == "z"){
      if(!(s->k2g_.serialize_status())){
        cout << "serialization enabled" << endl;
        s->k2g_.enableSerialization();
      } else{
        cout << "serialization disabled" << endl;
        s->k2g_.disableSerialization();
      }
    }
#endif
  }
}
