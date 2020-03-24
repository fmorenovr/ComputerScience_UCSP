#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/parula.h>
#include <igl/marching_tets.h>
#include <igl/winding_number.h>
#include <igl/slice.h>

#include <igl/parallel_for.h>

#include <igl/readMESH.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Sparse>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

Eigen::MatrixXd V, BC, Q;
Eigen::VectorXd W;
Eigen::MatrixXi T, F, G;
//Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor> F;

double slice_z = 0.5;
int order = 2;
double accuracy_scale = 2.0;

enum OverLayType{
  OVERLAY_NONE = 0,
  OVERLAY_INPUT = 1,
  OVERLAY_OUTPUT = 2,
  NUM_OVERLAY = 3,
} overlay = OVERLAY_NONE;

void update_visualization(igl::opengl::glfw::Viewer & viewer){
  Eigen::Vector4d plane(
    0,0,1,-((1-slice_z)*V.col(2).minCoeff()+slice_z*V.col(2).maxCoeff()));
  MatrixXd V_vis;
  MatrixXi F_vis;
  VectorXi J;
  {
    SparseMatrix<double> bary;
    // Value of plane's implicit function at all vertices
    const VectorXd IV = (V.col(0)*plane(0) + V.col(1)*plane(1) + V.col(2)*plane(2)).array() + plane(3);
    igl::marching_tets(V,T,IV,V_vis,F_vis,J,bary);
  }
  VectorXd W_vis;
  igl::slice(W,J,W_vis);
  MatrixXd C_vis;
  // color without normalizing
  igl::parula(W_vis,false,C_vis);
  const auto & append_mesh = [&C_vis,&F_vis,&V_vis](
    const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const RowVector3d & color) {
    F_vis.conservativeResize(F_vis.rows()+F.rows(),3);
    F_vis.bottomRows(F.rows()) = F.array()+V_vis.rows();
    V_vis.conservativeResize(V_vis.rows()+V.rows(),3);
    V_vis.bottomRows(V.rows()) = V;
    C_vis.conservativeResize(C_vis.rows()+F.rows(),3);
    C_vis.bottomRows(F.rows()).rowwise() = color;
  };
  switch(overlay) {
    case OVERLAY_INPUT:
      append_mesh(V,F,RowVector3d(1.,0.894,0.227));
      break;
    case OVERLAY_OUTPUT:
      append_mesh(V,G,RowVector3d(0.8,0.8,0.8));
      break;
    default:
      break;
  }
  viewer.data().clear();
  viewer.data().set_mesh(V_vis,F_vis);
  viewer.data().set_colors(C_vis);
  viewer.data().set_face_based(true);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod){
  switch(key){
    default:
      return false;
    case ' ':
      overlay = (OverLayType)((1+(int)overlay)%NUM_OVERLAY);
      break;
    case '.':
      slice_z = std::min(slice_z+0.01,0.99);
      break;
    case ',':
      slice_z = std::max(slice_z-0.01,0.01);
      break;
  }
  update_visualization(viewer);
  return true;
}
